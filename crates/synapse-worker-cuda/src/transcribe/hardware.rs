//! Opt-in hardware certification exercises the production LOAD and TRANSCRIBE
//! handlers on both Windows and Linux, including their multi-frame wire output.
use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{self, Cursor},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use synapse_core::{
    decode_decode_trace_frame, decode_transcript_frame, worker_framing_sync::read_json_frame,
    CUDA_TRANSCRIBE_KERNEL_REVISION, DEFAULT_MAX_FRAME_BYTES,
};
use synapse_parity::{
    manifest::Manifest,
    speech::{self, Artifact, SpeechObservation, SpeechProfile, SpeechRefusal},
};

struct Exchange {
    input: Cursor<Vec<u8>>,
    output: Vec<u8>,
    last: Instant,
    longest: Duration,
    progress: usize,
}
impl Read for Exchange {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        self.input.read(b)
    }
}
impl Write for Exchange {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if b.starts_with(b"{\"type\":\"PROGRESS\"") {
            self.longest = self.longest.max(self.last.elapsed());
            self.last = Instant::now();
            self.progress += 1;
            println!("completed-work progress {}", self.progress);
        }
        self.output.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn exchange(
    state: &WorkerState,
    id: &str,
    mel: &[f32],
    frames: usize,
    valid: usize,
    observe: bool,
) -> (SpeechObservation, usize, f64) {
    let mut input = Vec::new();
    write_frame(&mut input, &encode_f32_frame(mel), DEFAULT_MAX_FRAME_BYTES).unwrap();
    let mut stream = Exchange {
        input: Cursor::new(input),
        output: Vec::new(),
        last: Instant::now(),
        longest: Duration::ZERO,
        progress: 0,
    };
    serve_transcribe(
        &mut stream,
        DEFAULT_MAX_FRAME_BYTES,
        state,
        WorkerRequest::Transcribe {
            req_id: id.into(),
            model_ref: state.loaded.as_ref().unwrap().model_ref.clone(),
            mel_frames: frames,
            valid_mel_frames: valid,
            observe,
        },
    )
    .unwrap();
    let longest = stream.longest.max(stream.last.elapsed()).as_secs_f64();
    assert!(
        longest < 120.,
        "{id}: completed-work progress gap {longest}s"
    );
    let progress = stream.progress;
    let mut stream = Cursor::new(stream.output);
    let mut output = SpeechObservation::default();
    let mut stage = 0;
    loop {
        let response: WorkerResponse =
            read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
        match response {
            WorkerResponse::Progress { .. } => {
                assert_eq!(stage, 0);
            }
            WorkerResponse::EncoderOutput { frames, .. } => {
                assert!(observe);
                assert_eq!(stage, 0);
                stage = 1;
                let bytes = read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
                assert_eq!(bytes.len(), frames * 4096);
                output.encoder_output = Some(decode_f32_frame(&bytes).unwrap());
            }
            WorkerResponse::DecodeTrace { steps, .. } => {
                assert!(observe);
                assert_eq!(stage, 1);
                stage = 2;
                let bytes = read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
                let records = decode_decode_trace_frame(&bytes, steps).unwrap();
                output.trace = speech::decode_trace(&encode_decode_trace_frame(&records)).unwrap();
            }
            WorkerResponse::Transcript {
                valid_encoder_frames,
                count,
                ..
            } => {
                assert_eq!(stage, if observe { 2 } else { 0 });
                output.valid_encoder_frames = Some(valid_encoder_frames as u64);
                output.engine_started = true;
                let records = decode_transcript_frame(
                    &read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap(),
                    count,
                )
                .unwrap();
                output.triples = records
                    .iter()
                    .map(|t| [t.token_id, t.frame_index, t.duration])
                    .collect();
                break;
            }
            WorkerResponse::Err { code, .. } => {
                assert_eq!(stage, 0);
                assert_eq!(code, "duration_limit");
                output.refusal = Some(SpeechRefusal {
                    code,
                    details: [0, 0, valid.div_ceil(8) as u64, 0],
                });
                break;
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }
    assert_eq!(stream.position(), stream.get_ref().len() as u64);
    (output, progress, longest)
}

fn f32le(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn require_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
}
fn read_wav_input(root: &Path, id: &str, index: &Value) -> Vec<u8> {
    let row = &index[id];
    let bytes = std::fs::read(root.join(row["path"].as_str().unwrap())).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        row["sha256"].as_str().unwrap()
    );
    bytes
}

fn assert_same_decode(
    id: &str,
    check: &str,
    first: &SpeechObservation,
    second: &SpeechObservation,
) {
    assert_eq!(
        first.valid_encoder_frames, second.valid_encoder_frames,
        "{id}: {check}: valid encoder-frame counts differ"
    );
    let differing = first
        .triples
        .iter()
        .zip(&second.triples)
        .position(|(a, b)| a != b)
        .unwrap_or(first.triples.len().min(second.triples.len()));
    assert!(
        first.triples == second.triples,
        "{id}: {check}: first differing triple {differing}: {:?} vs {:?}",
        first.triples.get(differing),
        second.triples.get(differing)
    );
}

fn repeat_plain(
    state: &WorkerState,
    id: &str,
    mel: &[f32],
    frames: usize,
    valid: usize,
) -> (SpeechObservation, usize, f64) {
    let (first, progress, gap) = exchange(state, id, mel, frames, valid, false);
    let (second, second_progress, second_gap) = exchange(state, id, mel, frames, valid, false);
    assert!(first.refusal.is_none(), "{id}: repeatability input refused");
    assert!(second.refusal.is_none(), "{id}: repeated input refused");
    assert_same_decode(
        id,
        "nondeterminism with observe=false twice",
        &first,
        &second,
    );
    assert_eq!(progress, second_progress, "{id}: repeated progress differs");
    (first, progress, gap.max(second_gap))
}

fn load_state(manifest: &Value, profile_id: &str, maximum: u64) -> WorkerState {
    let mut state = WorkerState::default();
    let started = Instant::now();
    let loaded = load_transcribe(
        &mut state,
        "load-speech".into(),
        require_path("SYNAPSE_CUDA_SPEECH_PACKAGE")
            .to_string_lossy()
            .into_owned(),
        manifest["profiles"][profile_id]["converted_package_digest"]
            .as_str()
            .unwrap()
            .into(),
        "safetensors-package".into(),
        BTreeMap::from([
            ("profile".into(), profile_id.into()),
            ("operation".into(), "transcribe".into()),
            ("max_encoder_frames".into(), maximum.to_string()),
        ]),
        DEFAULT_MAX_FRAME_BYTES,
    );
    assert!(started.elapsed().as_secs() < 600);
    let WorkerResponse::Loaded {
        transcribe_revision: Some(revision),
        ..
    } = loaded
    else {
        panic!("{loaded:?}")
    };
    assert_eq!(revision.kernel_revision, CUDA_TRANSCRIBE_KERNEL_REVISION);
    assert_eq!(revision.decode_rule, synapse_speech::DECODE_RULE_REVISION);
    state
}

#[test]
#[ignore = "requires real NVIDIA hardware, pinned package and external tensor mirror; diagnostic only, not certification"]
fn cuda_transcribe_ladder_1023_is_repeatable_and_observation_read_only() {
    let parity = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity");
    let mirror = require_path("SYNAPSE_CUDA_SPEECH_MIRROR");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(parity.join("models.json")).unwrap()).unwrap();
    let document: Value = serde_json::from_slice(
        &std::fs::read(parity.join("fixtures/parakeet-tdt-0.6b-v3/reference.json")).unwrap(),
    )
    .unwrap();
    let id = "ladder-1023";
    let row = document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"].as_str() == Some(id))
        .unwrap();
    let artifact: Artifact = serde_json::from_value(row["log_mel"].clone()).unwrap();
    let mel = f32le(&artifact.read(&parity, &mirror).unwrap());
    let frames = row["mel_frames"].as_u64().unwrap() as usize;
    let valid = row["valid_mel_frames"].as_u64().unwrap() as usize;
    let state = load_state(&manifest, "parakeet-tdt-0.6b-v3.owned-cuda", 1023);
    let (plain, _, _) = repeat_plain(&state, id, &mel, frames, valid);
    let (observed, _, _) = exchange(&state, id, &mel, frames, valid, true);
    let (observed_again, _, _) = exchange(&state, id, &mel, frames, valid, true);
    assert_same_decode(id, "observation invariance", &plain, &observed);
    assert_same_decode(
        id,
        "nondeterminism with observe=true twice",
        &observed,
        &observed_again,
    );
    let encoder = observed.encoder_output.as_ref().unwrap();
    let encoder_again = observed_again.encoder_output.as_ref().unwrap();
    assert_eq!(encoder.len(), encoder_again.len());
    let differing = encoder
        .iter()
        .zip(encoder_again)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        differing.is_none(),
        "{id}: repeated encoder outputs differ at float index {differing:?}"
    );
    assert_eq!(observed.trace.len(), observed_again.trace.len());
    let logits = |step: &speech::TraceStep| {
        [
            step.token_logit1.to_bits(),
            step.token_logit2.to_bits(),
            step.duration_logit1.to_bits(),
            step.duration_logit2.to_bits(),
        ]
    };
    let differing = observed
        .trace
        .iter()
        .zip(&observed_again.trace)
        .position(|(a, b)| a != b || logits(a) != logits(b));
    assert!(
        differing.is_none(),
        "{id}: repeated joint traces differ at step {differing:?}"
    );
    println!("{id}: two plain and two observed executions are identical");
}

#[test]
#[ignore = "requires real NVIDIA hardware, pinned package, all prepared audio and external tensor mirror; see scripts/transcribe-certify.md"]
fn cuda_transcribe_certifies_tensor_trace_wer_and_observation_invariance() {
    let parity = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity");
    let mirror = require_path("SYNAPSE_CUDA_SPEECH_MIRROR");
    let audio = require_path("SYNAPSE_CUDA_SPEECH_AUDIO");
    let evidence = require_path("SYNAPSE_CUDA_SPEECH_EVIDENCE");
    std::fs::create_dir_all(&evidence).unwrap();
    let manifest = Manifest::load(&parity.join("models.json")).unwrap();
    let manifest_json: Value =
        serde_json::from_slice(&std::fs::read(parity.join("models.json")).unwrap()).unwrap();
    let profile_id = "parakeet-tdt-0.6b-v3.owned-cuda";
    let mut profile: SpeechProfile =
        serde_json::from_value(manifest_json["profiles"][profile_id]["transcribe"].clone())
            .unwrap();
    if let Ok(ms) = std::env::var("SYNAPSE_CUDA_SPEECH_MAX_MS") {
        profile.max_duration_ms = ms.parse().unwrap();
    }
    let maximum = speech::frame_counts(profile.max_duration_ms * 16).2;
    let fixtures = speech::load_transcribe_fixtures(&parity, &manifest).unwrap();
    assert!(
        fixtures
            .iter()
            .any(|f| f.id.starts_with("ladder-") && f.duration_ms == profile.max_duration_ms),
        "maximum must be an exact sealed ladder duration"
    );
    let index: Value =
        serde_json::from_slice(&std::fs::read(audio.join("inputs.json")).unwrap()).unwrap();
    let started = Instant::now();
    let state = load_state(&manifest_json, profile_id, maximum);
    let tokenizer = synapse_speech::Tokenizer::pinned();
    let frontend = synapse_speech::Frontend::new();
    let document: Value = serde_json::from_slice(
        &std::fs::read(parity.join("fixtures/parakeet-tdt-0.6b-v3/reference.json")).unwrap(),
    )
    .unwrap();
    let rows: BTreeMap<_, _> = document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (r["id"].as_str().unwrap(), r))
        .collect();
    let mut observations = BTreeMap::new();
    let mut invariant = 0;
    let mut longest = 0f64;
    let mut max_peak = 0;
    let mut largest_ladder = 0;
    for fixture in &fixtures {
        let row = rows[fixture.id.as_str()];
        let valid = row["valid_mel_frames"].as_u64().unwrap() as usize;
        let frames = row["mel_frames"].as_u64().unwrap() as usize;
        let over =
            fixture.duration_ms > profile.max_duration_ms || fixture.valid_encoder_frames > maximum;
        let mel = if over {
            vec![0.; frames * 128]
        } else if !row["log_mel"].is_null() {
            let artifact: Artifact = serde_json::from_value(row["log_mel"].clone()).unwrap();
            f32le(&artifact.read(&parity, &mirror).unwrap())
        } else {
            let wav = read_wav_input(&audio, &fixture.id, &index);
            let features = frontend.process_wav(&wav).unwrap();
            assert_eq!(features.mel_frames, frames);
            assert_eq!(features.valid_mel_frames, valid);
            features.values
        };
        let (mut observed, progress, gap) =
            exchange(&state, &fixture.id, &mel, frames, valid, true);
        longest = longest.max(gap);
        if over {
            let refusal = observed
                .refusal
                .as_mut()
                .expect("above-limit clip was executed");
            refusal.details = [
                fixture.duration_ms,
                profile.max_duration_ms,
                fixture.valid_encoder_frames,
                maximum,
            ];
            assert_eq!(progress, 0);
        } else {
            assert!(observed.refusal.is_none(), "within-limit fixture refused");
            assert_eq!(progress, 24 + fixture.valid_encoder_frames as usize / 125);
            observed.text = tokenizer
                .transcript(
                    &observed
                        .triples
                        .iter()
                        .map(|t| t.map(|v| u32::try_from(v).unwrap()))
                        .collect::<Vec<_>>(),
                    fixture.duration_ms as f64,
                )
                .unwrap()
                .text;
            if fixture.encoder.is_some() {
                let (plain, plain_progress, plain_gap) =
                    repeat_plain(&state, &fixture.id, &mel, frames, valid);
                assert_same_decode(&fixture.id, "observation invariance", &observed, &plain);
                assert_eq!(progress, plain_progress);
                longest = longest.max(plain_gap);
                invariant += 1;
            }
            if fixture.id.starts_with("ladder-") {
                largest_ladder = largest_ladder.max(fixture.duration_ms);
            }
        }
        if let Some(super::super::LoadedModel {
            engine: super::super::LoadedEngine::Transcribe { engine, .. },
            ..
        }) = &state.loaded
        {
            max_peak = max_peak.max(engine.peak_device_bytes());
        }
        println!(
            "{} frames={} progress={} gap={gap:.3}s peak={max_peak}",
            fixture.id, fixture.valid_encoder_frames, progress
        );
        observations.insert(fixture.id.clone(), observed);
    }
    let report = speech::evaluate_transcribe(
        &profile,
        &fixtures,
        &observations,
        &parity,
        &mirror,
        "duration_limit",
    )
    .unwrap();
    let summary = json!({"profile":profile_id,"kernel_revision":CUDA_TRANSCRIBE_KERNEL_REVISION,"decode_rule":synapse_speech::DECODE_RULE_REVISION,
        "graded":report.graded.len(),"expected_refusals":report.expected_refusals,"margin_exempt":report.margin_exempt,"split_wer":report.split_wer,
        "tensor_observation_invariance":invariant,"tensor_repeatability":invariant,"max_duration_ms":profile.max_duration_ms,"largest_certified_ladder_ms":largest_ladder,
        "peak_device_used_bytes":max_peak,"longest_progress_gap_s":longest,"elapsed_s":started.elapsed().as_secs_f64()});
    std::fs::write(
        evidence.join("cuda-transcribe-summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    println!("{summary}");
}
