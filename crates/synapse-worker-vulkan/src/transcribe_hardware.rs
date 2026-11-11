//! Opt-in runs use the production encoder, joint and framed response writer.
use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    io::Cursor,
    path::PathBuf,
    time::{Duration, Instant},
};
use synapse_core::{decode_decode_trace_frame, decode_transcript_frame};
use synapse_parity::speech::{self, Artifact, SpeechObservation, SpeechProfile, SpeechRefusal};

struct Exchange {
    output: Vec<u8>,
    last: Instant,
    longest: Duration,
    progress: usize,
}
impl Write for Exchange {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.starts_with(b"{\"type\":\"PROGRESS\"") {
            self.longest = self.longest.max(self.last.elapsed());
            self.last = Instant::now();
            self.progress += 1;
            println!("completed-work progress {}", self.progress);
        }
        self.output.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn exchange(
    model: &mut Loaded,
    id: &str,
    mel: &[f32],
    frames: usize,
    valid: usize,
    observe: bool,
) -> (SpeechObservation, usize, f64) {
    let mut stream = Exchange {
        output: Vec::new(),
        last: Instant::now(),
        longest: Duration::ZERO,
        progress: 0,
    };
    let request = WorkerRequest::Transcribe {
        req_id: id.into(),
        model_ref: model.model_ref.clone(),
        mel_frames: frames,
        valid_mel_frames: valid,
        observe,
    };
    let raw = synapse_core::encode_f32_frame(mel);
    request.validate_raw_frame(&raw).unwrap();
    let model_ref = model.model_ref.clone();
    serve_transcribe(
        &mut stream,
        DEFAULT_MAX_FRAME_BYTES,
        Some(model),
        id,
        &model_ref,
        frames,
        valid,
        &raw,
        observe,
    )
    .unwrap();
    let gap = stream.longest.max(stream.last.elapsed()).as_secs_f64();
    assert!(gap < 120., "{id}: completed-work progress gap {gap}s");
    let progress = stream.progress;
    let mut input = Cursor::new(stream.output);
    let mut output = SpeechObservation::default();
    let mut stage = 0;
    loop {
        let response: WorkerResponse =
            read_json_frame(&mut input, DEFAULT_MAX_FRAME_BYTES).unwrap();
        match response {
            WorkerResponse::Progress { completed, .. } => {
                assert_eq!(stage, 0);
                assert!(completed > 0);
            }
            WorkerResponse::EncoderOutput { frames, .. } => {
                assert!(observe);
                assert_eq!(stage, 0);
                stage = 1;
                let raw = read_frame(&mut input, DEFAULT_MAX_FRAME_BYTES).unwrap();
                assert_eq!(raw.len(), frames * 4096);
                output.encoder_output = Some(synapse_core::decode_f32_frame(&raw).unwrap());
            }
            WorkerResponse::DecodeTrace { steps, .. } => {
                assert!(observe);
                assert_eq!(stage, 1);
                stage = 2;
                let raw = read_frame(&mut input, DEFAULT_MAX_FRAME_BYTES).unwrap();
                decode_decode_trace_frame(&raw, steps).unwrap();
                output.trace = speech::decode_trace(&raw).unwrap();
            }
            WorkerResponse::Transcript {
                valid_encoder_frames,
                count,
                ..
            } => {
                assert_eq!(stage, if observe { 2 } else { 0 });
                output.valid_encoder_frames = Some(valid_encoder_frames as u64);
                output.engine_started = true;
                output.triples = decode_transcript_frame(
                    &read_frame(&mut input, DEFAULT_MAX_FRAME_BYTES).unwrap(),
                    count,
                )
                .unwrap()
                .iter()
                .map(|r| [r.token_id, r.frame_index, r.duration])
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
    assert_eq!(input.position(), input.get_ref().len() as u64);
    (output, progress, gap)
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
}

fn load_hardware() -> Loaded {
    let profile_id = "parakeet-tdt-0.6b-v3.owned-vulkan";
    let package = env_path("SYNAPSE_VULKAN_SPEECH_PACKAGE");
    let manifest = crate::manifest();
    let profile = &manifest.profiles[profile_id];
    let digest = profile.converted_package_digest.as_ref().unwrap();
    let config: BTreeMap<String, String> = BTreeMap::from([
        ("profile".into(), profile_id.into()),
        ("operation".into(), "transcribe".into()),
    ]);
    #[cfg(all(target_os = "macos", feature = "moltenvk-diagnostic"))]
    {
        // This constructor is deliberately test-only. Production LOAD continues
        // to reject Apple Vulkan adapters, so no macOS Vulkan lane is exposed.
        let required = requirements(&manifest, Some(&profile.model)).unwrap();
        let prepared =
            crate::runtime::prepare_moltenvk_parity(&env_path("SYNAPSE_MOLTENVK_LOADER"), required)
                .unwrap();
        let bytes = std::fs::read(package).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            super::digest(digest)
        );
        synapse_parity::safetensors::verify_package_structure(&bytes, profile_id).unwrap();
        let (header, data) = synapse_parity::safetensors::split_file(&bytes).unwrap();
        let header = parse_header_json(header).unwrap();
        let engine = crate::runtime::transcribe::Engine::load(
            &manifest.models[&profile.model],
            profile,
            prepared,
            required,
            &header,
            data,
        )
        .unwrap();
        let _ = config;
        Loaded {
            model_ref: digest.clone(),
            operation: Operation::Transcribe,
            dims: 0,
            engine: LoadedEngine::Speech(Box::new(engine)),
        }
    }
    #[cfg(not(all(target_os = "macos", feature = "moltenvk-diagnostic")))]
    load(
        &package.to_string_lossy(),
        digest,
        &config,
        DEFAULT_MAX_FRAME_BYTES,
        crate::enumerate,
    )
    .unwrap_or_else(|code| panic!("LOAD failed: {code}"))
}

#[test]
#[ignore = "requires NVIDIA hardware or explicit MoltenVK diagnostic, pinned package, audio and tensor mirror; see scripts/transcribe-certify.md"]
fn vulkan_transcribe_certifies_tensor_trace_wer_and_observation_invariance() {
    let parity = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity");
    let mirror = env_path("SYNAPSE_VULKAN_SPEECH_MIRROR");
    let audio = env_path("SYNAPSE_VULKAN_SPEECH_AUDIO");
    let evidence = env_path("SYNAPSE_VULKAN_SPEECH_EVIDENCE");
    let filter = std::env::var("SYNAPSE_VULKAN_SPEECH_FILTER").ok();
    std::fs::create_dir_all(&evidence).unwrap();
    let manifest = crate::manifest();
    let profile_id = "parakeet-tdt-0.6b-v3.owned-vulkan";
    let document: Value =
        serde_json::from_slice(&std::fs::read(parity.join("models.json")).unwrap()).unwrap();
    let profile: SpeechProfile =
        serde_json::from_value(document["profiles"][profile_id]["transcribe"].clone()).unwrap();
    let maximum = speech::frame_counts(profile.max_duration_ms * 16).2;
    let fixtures = speech::load_transcribe_fixtures(&parity, &manifest).unwrap();
    assert!(fixtures
        .iter()
        .any(|f| f.id.starts_with("ladder-") && f.duration_ms == profile.max_duration_ms));
    assert!(fixtures
        .iter()
        .filter(|f| !f.id.starts_with("ladder-"))
        .all(|f| f.duration_ms <= profile.max_duration_ms));
    let fixtures: Vec<_> = fixtures
        .into_iter()
        .filter(|f| {
            filter
                .as_ref()
                .is_none_or(|prefix| f.id.starts_with(prefix))
        })
        .collect();
    assert!(!fixtures.is_empty(), "filter selected no fixtures");
    let index: Value =
        serde_json::from_slice(&std::fs::read(audio.join("inputs.json")).unwrap()).unwrap();
    let started = Instant::now();
    let mut loaded = load_hardware();
    assert!(started.elapsed().as_secs() < 600);
    let revision = loaded_response("load".into(), &loaded);
    assert!(
        matches!(revision,WorkerResponse::Loaded {transcribe_revision:Some(r),..} if r.kernel_revision==synapse_core::worker_engine_names::VULKAN_TRANSCRIBE_KERNEL_REVISION && r.decode_rule==synapse_speech::DECODE_RULE_REVISION)
    );
    let tokenizer = synapse_speech::Tokenizer::pinned();
    let frontend = synapse_speech::Frontend::new();
    let reference: Value = serde_json::from_slice(
        &std::fs::read(parity.join("fixtures/parakeet-tdt-0.6b-v3/reference.json")).unwrap(),
    )
    .unwrap();
    let rows: BTreeMap<_, _> = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (r["id"].as_str().unwrap(), r))
        .collect();
    let mut observations = BTreeMap::new();
    let (mut invariant, mut longest, mut largest_ladder) = (0, 0f64, 0);
    for fixture in &fixtures {
        let row = rows[fixture.id.as_str()];
        let frames = row["mel_frames"].as_u64().unwrap() as usize;
        let valid = row["valid_mel_frames"].as_u64().unwrap() as usize;
        let over =
            fixture.duration_ms > profile.max_duration_ms || fixture.valid_encoder_frames > maximum;
        let mel = if over {
            vec![0.; frames * 128]
        } else if !row["log_mel"].is_null() {
            let artifact: Artifact = serde_json::from_value(row["log_mel"].clone()).unwrap();
            synapse_core::decode_f32_frame(&artifact.read(&parity, &mirror).unwrap()).unwrap()
        } else {
            let bytes =
                std::fs::read(audio.join(index[&fixture.id]["path"].as_str().unwrap())).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                index[&fixture.id]["sha256"].as_str().unwrap()
            );
            let features = frontend.process_wav(&bytes).unwrap();
            assert_eq!(features.mel_frames, frames);
            assert_eq!(features.valid_mel_frames, valid);
            features.values
        };
        let (mut output, progress, gap) =
            exchange(&mut loaded, &fixture.id, &mel, frames, valid, true);
        longest = longest.max(gap);
        if over {
            output
                .refusal
                .as_mut()
                .expect("over-limit clip executed")
                .details = [
                fixture.duration_ms,
                profile.max_duration_ms,
                fixture.valid_encoder_frames,
                maximum,
            ];
            assert_eq!(progress, 0);
        } else {
            assert_eq!(progress, 24 + fixture.valid_encoder_frames as usize / 125);
            output.text = tokenizer
                .transcript(
                    &output
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
                    exchange(&mut loaded, &fixture.id, &mel, frames, valid, false);
                assert_eq!(output.valid_encoder_frames, plain.valid_encoder_frames);
                assert_eq!(output.triples, plain.triples);
                assert_eq!(progress, plain_progress);
                longest = longest.max(plain_gap);
                invariant += 1;
            }
            if fixture.id.starts_with("ladder-") {
                largest_ladder = largest_ladder.max(fixture.duration_ms);
            }
        }
        println!(
            "{} frames={} progress={progress} gap={gap:.3}s",
            fixture.id, fixture.valid_encoder_frames
        );
        observations.insert(fixture.id.clone(), output);
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
    let LoadedEngine::Speech(engine) = &loaded.engine else {
        panic!("wrong engine");
    };
    let peak = engine.device_bytes();
    assert!(
        peak <= manifest.profiles[profile_id]
            .vulkan_min_device_local_bytes
            .unwrap()
    );
    let summary = json!({"profile":profile_id,"complete":filter.is_none(),"filter":filter,
        "diagnostic_moltenvk":cfg!(all(target_os="macos",feature="moltenvk-diagnostic")),
        "kernel_revision":synapse_core::worker_engine_names::VULKAN_TRANSCRIBE_KERNEL_REVISION,
        "decode_rule":synapse_speech::DECODE_RULE_REVISION,"graded":report.graded.len(),"expected_refusals":report.expected_refusals,
        "margin_exempt":report.margin_exempt,"split_wer":report.split_wer,"tensor_observation_invariance":invariant,
        "max_duration_ms":profile.max_duration_ms,"largest_certified_ladder_ms":largest_ladder,
        "arena_driver_allocation_bytes":peak,"longest_progress_gap_s":longest,"elapsed_s":started.elapsed().as_secs_f64()});
    std::fs::write(
        evidence.join("vulkan-transcribe-summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    println!("{summary}");
}
