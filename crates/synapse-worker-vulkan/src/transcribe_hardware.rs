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

fn max_abs(a: &[f32], b: &[f32]) -> Option<f32> {
    (a.len() == b.len()).then(|| {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max)
    })
}

fn trace_json(step: Option<&speech::TraceStep>) -> Value {
    step.map_or(Value::Null,|s|json!({"frame_index":s.frame_index,
        "token_ids":[s.token_top1,s.token_top2],"token_logits":[s.token_logit1,s.token_logit2],
        "token_margin":s.token_logit1-s.token_logit2,
        "duration_ids":[s.duration_top1,s.duration_top2],"duration_logits":[s.duration_logit1,s.duration_logit2],
        "duration_margin":s.duration_logit1-s.duration_logit2}))
}

fn first_difference(
    reference: &[speech::TraceStep],
    actual: &[speech::TraceStep],
) -> Option<usize> {
    (0..reference.len().max(actual.len())).find(|i| match (reference.get(*i), actual.get(*i)) {
        (Some(r), Some(a)) => {
            r.frame_index != a.frame_index
                || r.token_top1 != a.token_top1
                || r.duration_top1 != a.duration_top1
        }
        _ => true,
    })
}

fn trace_finding(
    id: &str,
    reference: &[speech::TraceStep],
    actual: &[speech::TraceStep],
) -> Option<Value> {
    first_difference(reference,actual).map(|index| {
        let (r,a) = (reference.get(index),actual.get(index));
        let mut heads = Vec::new();
        if let (Some(r),Some(a)) = (r,a) {
            if r.token_top1!=a.token_top1 {
                heads.push(json!({"head":"token","reference_margin":r.token_logit1-r.token_logit2,
                    "reference_pair":[r.token_top1,r.token_top2],"reference_logits":[r.token_logit1,r.token_logit2],
                    "engine_pair":[a.token_top1,a.token_top2],"engine_logits":[a.token_logit1,a.token_logit2]}));
            }
            if r.duration_top1!=a.duration_top1 {
                heads.push(json!({"head":"duration","reference_margin":r.duration_logit1-r.duration_logit2,
                    "reference_pair":[r.duration_top1,r.duration_top2],"reference_logits":[r.duration_logit1,r.duration_logit2],
                    "engine_pair":[a.duration_top1,a.duration_top2],"engine_logits":[a.duration_logit1,a.duration_logit2]}));
            }
        }
        json!({"fixture":id,"step":index,"reference":trace_json(r),"engine":trace_json(a),"heads":heads,
            "same_frame":r.zip(a).is_some_and(|(r,a)|r.frame_index==a.frame_index)})
    })
}

fn split_wer_rates(
    fixtures: &[speech::SpeechFixture],
    observations: &BTreeMap<String, SpeechObservation>,
) -> BTreeMap<String, [f64; 2]> {
    let mut counts: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    for fixture in fixtures {
        if let (Some(split), Some(truth)) = (&fixture.split, &fixture.ground_truth) {
            let truth = speech::normalize_words(truth);
            let row = counts.entry(split.clone()).or_default();
            row[0] += truth.len();
            row[1] += speech::word_edits(&truth, &speech::normalize_words(&fixture.text));
            row[2] += speech::word_edits(
                &truth,
                &speech::normalize_words(&observations[&fixture.id].text),
            );
        }
    }
    counts
        .into_iter()
        .map(|(split, [words, reference, actual])| {
            assert!(words > 0);
            (
                split,
                [
                    reference as f64 / words as f64,
                    actual as f64 / words as f64,
                ],
            )
        })
        .collect()
}

#[test]
fn transcribe_margin_findings_keep_head_pairs_and_reference_margin() {
    let record = synapse_core::DecodeTraceRecord {
        frame_index: 417,
        token_top1: 7990,
        token_top2: 497,
        duration_top1: 4,
        duration_top2: 3,
        token_logit1: -3.620,
        token_logit2: -7.889,
        duration_logit1: 11.35513,
        duration_logit2: 11.35067,
        flags: 1,
    };
    let reference =
        speech::decode_trace(&synapse_core::encode_decode_trace_frame(&[record])).unwrap();
    let mut actual = reference.clone();
    actual[0].duration_top1 = 3;
    actual[0].duration_top2 = 4;
    let finding = trace_finding("fixture", &reference, &actual).unwrap();
    assert_eq!(finding["step"], 0);
    assert_eq!(finding["heads"].as_array().unwrap().len(), 1);
    assert_eq!(finding["heads"][0]["head"], "duration");
    assert_eq!(finding["heads"][0]["reference_pair"], json!([4, 3]));
    assert_eq!(finding["heads"][0]["engine_pair"], json!([3, 4]));
    let margin = finding["heads"][0]["reference_margin"].as_f64().unwrap();
    assert!((margin - 0.00446).abs() < 0.000002);
    assert!(margin > 0.001);
    assert!(trace_finding("same", &reference, &reference).is_none());
    assert_eq!(
        trace_finding("short", &reference, &[]).unwrap()["same_frame"],
        false
    );
}

#[test]
fn transcribe_wer_findings_preserve_failing_split_rates() {
    let parity = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity");
    let mut fixture = speech::load_transcribe_fixtures(&parity, &crate::manifest())
        .unwrap()
        .remove(0);
    fixture.split = Some("test".into());
    fixture.ground_truth = Some("a b".into());
    fixture.text = "a b".into();
    let observation = SpeechObservation {
        text: "a x".into(),
        ..Default::default()
    };
    let observations = BTreeMap::from([(fixture.id.clone(), observation)]);
    assert_eq!(
        split_wer_rates(&[fixture], &observations)["test"],
        [0., 0.5]
    );
}

#[allow(clippy::too_many_arguments)]
fn diagnose_trace_failure(
    loaded: &mut Loaded,
    fixture: &speech::SpeechFixture,
    mel: &[f32],
    frames: usize,
    valid: usize,
    actual: &SpeechObservation,
    reference_trace: &[speech::TraceStep],
    sealed_encoder: Option<&[f32]>,
    evidence: &std::path::Path,
) -> Result<()> {
    let index = first_difference(reference_trace, &actual.trace).unwrap_or(0);
    let encoder = actual.encoder_output.as_ref().unwrap();
    let frame = actual
        .trace
        .get(index)
        .map_or(0, |s| s.frame_index.max(0) as usize);
    let mut report = json!({"fixture":fixture.id,"step":index,"reference":trace_json(reference_trace.get(index)),"actual":trace_json(actual.trace.get(index))});
    let LoadedEngine::Speech(engine) = &loaded.engine else {
        panic!("wrong engine");
    };
    let projected = engine.debug_projection().to_vec();
    std::fs::write(
        evidence.join(format!("{}-encoder.f32le", fixture.id)),
        synapse_core::encode_f32_frame(encoder),
    )?;
    std::fs::write(
        evidence.join(format!("{}-projected.f32le", fixture.id)),
        synapse_core::encode_f32_frame(&projected),
    )?;

    // Repeat only a failing fixture. This distinguishes a stable numerical
    // difference from uninitialized/reused GPU memory or observation changes.
    let (repeat, _, _) = exchange(loaded, &fixture.id, mel, frames, valid, true);
    let (plain, _, _) = exchange(loaded, &fixture.id, mel, frames, valid, false);
    let repeated_encoder = repeat.encoder_output.as_ref().unwrap();
    report["repeat_encoder_max_abs"] = json!(max_abs(encoder, repeated_encoder));
    report["repeat_encoder_bit_identical"] = json!(
        encoder.len() == repeated_encoder.len()
            && encoder
                .iter()
                .zip(repeated_encoder)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    report["repeat_triples_identical"] = json!(actual.triples == repeat.triples);
    report["plain_triples_identical"] = json!(actual.triples == plain.triples);
    report["repeat_step"] = trace_json(repeat.trace.get(index));
    report["repeat_valid_frames_identical"] = json!(
        actual.valid_encoder_frames == repeat.valid_encoder_frames
            && actual.valid_encoder_frames == plain.valid_encoder_frames
    );

    let debug_encoder = std::env::var_os("SYNAPSE_VULKAN_SPEECH_DEBUG_ENCODER")
        .map(|p| -> Result<Vec<f32>> {
            let path = PathBuf::from(p);
            let bytes = std::fs::read(&path)?;
            let name = path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .split('.')
                .next()
                .unwrap();
            anyhow::ensure!(
                name.len() == 64 && format!("{:x}", Sha256::digest(&bytes)) == name,
                "debug encoder hash mismatch"
            );
            Ok(synapse_core::decode_f32_frame(&bytes)?)
        })
        .transpose()?;
    let reference_encoder = sealed_encoder.or(debug_encoder.as_deref());
    if let Some(reference_encoder) = reference_encoder {
        anyhow::ensure!(
            reference_encoder.len() == encoder.len(),
            "debug encoder frame count mismatch"
        );
        report["reference_encoder_source"] = json!(if sealed_encoder.is_some() {
            "sealed"
        } else {
            "scratch debug reference; not a seal"
        });
        report["encoder_reference_max_abs"] = json!(max_abs(encoder, reference_encoder));
        if frame * 1024 + 1024 <= encoder.len() {
            report["encoder_reference_frame_max_abs"] = json!(max_abs(
                &encoder[frame * 1024..(frame + 1) * 1024],
                &reference_encoder[frame * 1024..(frame + 1) * 1024]
            ));
        }
        let LoadedEngine::Speech(engine) = &mut loaded.engine else {
            panic!("wrong engine");
        };
        // These CPU diagnostics use the package's exact fp16 projector weights
        // with f64 dot products. They do not change production computations.
        let exact_actual = engine.debug_project_f64(encoder)?;
        let exact_reference = engine.debug_project_f64(reference_encoder)?;
        report["projection_gpu_vs_f64_actual_encoder_max_abs"] =
            json!(max_abs(&projected, &exact_actual));
        report["projection_gpu_vs_f64_reference_encoder_max_abs"] =
            json!(max_abs(&projected, &exact_reference));
        if frame * 640 + 640 <= projected.len() {
            report["projection_gpu_vs_f64_actual_frame_max_abs"] = json!(max_abs(
                &projected[frame * 640..(frame + 1) * 640],
                &exact_actual[frame * 640..(frame + 1) * 640]
            ));
            report["projection_gpu_vs_f64_reference_frame_max_abs"] = json!(max_abs(
                &projected[frame * 640..(frame + 1) * 640],
                &exact_reference[frame * 640..(frame + 1) * 640]
            ));
        }
        let replay_actual =
            engine.debug_replay_projection(exact_actual, fixture.valid_encoder_frames as usize)?;
        let replay_reference = engine
            .debug_replay_projection(exact_reference, fixture.valid_encoder_frames as usize)?;
        report["f64_projected_actual_replay_step"] = json!(replay_actual.trace.get(index));
        report["f64_projected_reference_replay_step"] = json!(replay_reference.trace.get(index));
    }
    std::fs::write(
        evidence.join("vulkan-trace-failure.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("trace failure diagnostics: {report}");
    Ok(())
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
    let mut graded = Vec::new();
    let mut expected_refusals = Vec::new();
    let mut margin_exempt = Vec::new();
    let mut trace_failures = Vec::new();
    let mut trace_findings = Vec::new();
    let mut encoder_failures = Vec::new();
    let mut diagnosed = false;
    let mut findings_log =
        std::fs::File::create(evidence.join("vulkan-fixture-findings.jsonl")).unwrap();
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
            expected_refusals.push(fixture.id.clone());
        } else {
            graded.push(fixture.id.clone());
            assert_eq!(progress, 24 + fixture.valid_encoder_frames as usize / 125);
            assert_eq!(
                output.valid_encoder_frames,
                Some(fixture.valid_encoder_frames)
            );
            let encoder = output
                .encoder_output
                .as_ref()
                .expect("missing encoder observation");
            assert_eq!(encoder.len(), fixture.valid_encoder_frames as usize * 1024);
            assert!(encoder.iter().all(|v| v.is_finite()));
            // Keep every fixture's first-decision finding, including failures.
            // The final evaluator still enforces the numerical gates.
            let reference_encoder = fixture.encoder.as_ref().map(|artifact| {
                synapse_core::decode_f32_frame(&artifact.read(&parity, &mirror).unwrap()).unwrap()
            });
            let mut encoder_ok = true;
            let cosine =
                reference_encoder.as_ref().and_then(|reference| {
                    match speech::grade_encoder(
                        reference,
                        encoder,
                        fixture.valid_encoder_frames as usize,
                    ) {
                        Ok(cosine) => Some(cosine),
                        Err(e) => {
                            encoder_ok = false;
                            encoder_failures
                                .push(json!({"fixture":fixture.id,"error":e.to_string()}));
                            None
                        }
                    }
                });
            let reference_trace =
                speech::decode_trace(&fixture.trace.read(&parity, &mirror).unwrap()).unwrap();
            let trace_grade = speech::grade_trace(
                &reference_trace,
                &output.trace,
                &output.triples,
                fixture.valid_encoder_frames,
                profile.tau_margin.unwrap(),
            );
            let mut finding = trace_finding(&fixture.id, &reference_trace, &output.trace);
            let trace_ok = trace_grade.is_ok();
            match trace_grade {
                Ok(exempt) => {
                    if exempt {
                        margin_exempt.push(fixture.id.clone());
                    }
                    if let Some(finding) = &mut finding {
                        finding["margin_exempt"] = json!(exempt);
                    }
                }
                Err(e) => {
                    let mut failure = finding
                        .clone()
                        .unwrap_or_else(|| json!({"fixture":fixture.id}));
                    failure["error"] = json!(e.to_string());
                    trace_failures.push(failure.clone());
                    finding = Some(failure);
                    if !diagnosed
                        && (filter.is_some()
                            || std::env::var_os("SYNAPSE_VULKAN_SPEECH_DEBUG_ENCODER").is_some())
                    {
                        diagnosed = true;
                        if let Err(diagnostic_error) = diagnose_trace_failure(
                            &mut loaded,
                            fixture,
                            &mel,
                            frames,
                            valid,
                            &output,
                            &reference_trace,
                            reference_encoder.as_deref(),
                            &evidence,
                        ) {
                            println!("trace diagnostic failed: {diagnostic_error}");
                        }
                    }
                }
            }
            if let Some(finding) = &finding {
                trace_findings.push(finding.clone());
            }
            writeln!(findings_log,"{}",json!({"fixture":fixture.id,"encoder_min_cosine":cosine,"encoder_ok":encoder_ok,"trace_ok":trace_ok,"trace_finding":finding})).unwrap();
            println!(
                "{} encoder_min_cosine={cosine:?} trace_ok={trace_ok}",
                fixture.id
            );
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
            if fixture.id.starts_with("ladder-") && trace_ok && encoder_ok {
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
    );
    let passed = report.is_ok();
    let split_wer = split_wer_rates(&fixtures, &observations);
    let wer_failures: Vec<_> = split_wer.iter().filter(|(_,rates)|rates[1]-rates[0]>0.005).map(|(split,rates)|json!({"split":split,"reference":rates[0],"engine":rates[1],"delta":rates[1]-rates[0]})).collect();
    let LoadedEngine::Speech(engine) = &loaded.engine else {
        panic!("wrong engine");
    };
    let peak = engine.device_bytes();
    assert!(
        peak <= manifest.profiles[profile_id]
            .vulkan_min_device_local_bytes
            .unwrap()
    );
    let summary = json!({"profile":profile_id,"complete":filter.is_none(),"passed":passed,"filter":filter,
        "diagnostic_moltenvk":cfg!(all(target_os="macos",feature="moltenvk-diagnostic")),
        "kernel_revision":synapse_core::worker_engine_names::VULKAN_TRANSCRIBE_KERNEL_REVISION,
        "decode_rule":synapse_speech::DECODE_RULE_REVISION,"graded":graded.len(),"expected_refusals":expected_refusals,
        "margin_exempt":margin_exempt,"trace_findings":trace_findings,"trace_failures":trace_failures,"encoder_failures":encoder_failures,
        "split_wer":split_wer,"wer_failures":wer_failures,"evaluation_error":report.err().map(|e|e.to_string()),"tensor_observation_invariance":invariant,
        "max_duration_ms":profile.max_duration_ms,"largest_passed_ladder_ms":largest_ladder,"largest_certified_ladder_ms":if passed && filter.is_none() {Some(largest_ladder)} else {None},
        "arena_driver_allocation_bytes":peak,"longest_progress_gap_s":longest,"elapsed_s":started.elapsed().as_secs_f64()});
    std::fs::write(
        evidence.join("vulkan-transcribe-summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    println!("{summary}");
    assert!(
        passed,
        "numerical certification failed; complete findings and split WER were written to {}",
        evidence.display()
    );
}
