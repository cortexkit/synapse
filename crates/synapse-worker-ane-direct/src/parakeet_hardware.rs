//! Opt-in certification, run only by the hardware owner holding the machine-wide Neural Engine lock.
use super::*;
use anyhow::Context;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;
use synapse_parity::{manifest::Manifest, speech};

const PROFILE: &str = "parakeet-tdt-0.6b-v3.ane-direct-worker";

fn env_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("{name} must name the pinned asset location"))
}

fn lane_lock() -> Result<File> {
    ensure!(
        std::env::var_os("TMPDIR").is_none(),
        "unset TMPDIR before ANE certification"
    );
    // Use the same path and nonblocking flock as the production supervisor.
    // No configurable lock path is allowed: a private lock would permit overlap.
    let path = env_path("HOME")?.join("Library/Caches/ck-synapse/ane-direct.lock");
    std::fs::create_dir_all(path.parent().unwrap())?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)?;
    lock.try_lock()
        .context("ane_lane_busy: another ANE client holds the machine-wide lock")?;
    Ok(lock)
}

fn f32le(bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(synapse_core::decode_f32_frame(bytes)?)
}

fn artifact(value: &Value, parity: &Path, mirror: &Path) -> Result<Vec<u8>> {
    let artifact: speech::Artifact = serde_json::from_value(value.clone())?;
    Ok(artifact.read(parity, mirror)?)
}

fn fixture_features(
    case: &Value,
    parity: &Path,
    mirror: &Path,
    wavs: &Path,
    inputs: &Value,
    frontend: &synapse_speech::Frontend,
) -> Result<(Vec<f32>, usize, usize)> {
    if !case["log_mel"].is_null() {
        return Ok((
            f32le(&artifact(&case["log_mel"], parity, mirror)?)?,
            case["mel_frames"].as_u64().context("mel_frames")? as usize,
            case["valid_mel_frames"]
                .as_u64()
                .context("valid_mel_frames")? as usize,
        ));
    }
    let id = case["id"].as_str().context("fixture id")?;
    let input = &inputs[id];
    let relative = input["path"]
        .as_str()
        .context("prepared WAV path missing")?;
    ensure!(
        Path::new(relative)
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_))),
        "prepared WAV path must be relative"
    );
    let bytes = std::fs::read(wavs.join(relative))?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes)) == input["sha256"].as_str().unwrap_or_default(),
        "prepared WAV checksum mismatch: {id}"
    );
    let features = frontend.process_wav(&bytes)?;
    ensure!(
        features.samples_16khz as u64 == case["sample_count"].as_u64().context("sample_count")?,
        "prepared WAV sample count mismatch: {id}"
    );
    Ok((
        features.values,
        features.mel_frames,
        features.valid_mel_frames,
    ))
}

/// Every interval starts at the preceding completed operation, not at a timer.
#[derive(Default)]
struct Progress {
    previous: Option<Instant>,
    maximum: f64,
    count: usize,
}
impl Progress {
    fn start() -> Self {
        Self {
            previous: Some(Instant::now()),
            ..Self::default()
        }
    }
    fn completed(&mut self) -> Result<()> {
        self.maximum = self
            .maximum
            .max(self.previous.unwrap().elapsed().as_secs_f64());
        self.previous = Some(Instant::now());
        self.count += 1;
        Ok(())
    }
    fn finish(&mut self, ceiling: f64) -> Result<()> {
        self.maximum = self
            .maximum
            .max(self.previous.unwrap().elapsed().as_secs_f64());
        ensure!(
            self.maximum < ceiling,
            "completed-work progress gap {} >= {ceiling}s",
            self.maximum
        );
        Ok(())
    }
}

fn certify(summary: &mut Value) -> Result<()> {
    let _lock = lane_lock()?;
    crate::worker::private_api()?;
    let parity = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity");
    let snapshot = env_path("ANE_PARAKEET_SNAPSHOT")?;
    let package = env_path("ANE_PARAKEET_PACKAGE")?;
    let manifest = Manifest::load(&parity.join("models.json"))?;
    if std::env::var("ANE_PARAKEET_PROBE").as_deref() == Ok("1") {
        return compile_probes(summary, &manifest, &snapshot, &package);
    }
    let mirror = env_path("ANE_PARAKEET_MIRROR")?;
    let wavs = env_path("ANE_PARAKEET_WAVS")?;
    let fixtures = speech::load_transcribe_fixtures(&parity, &manifest)?;
    // The loader verifies this document against the fixture index and manifest.
    let document: Value = serde_json::from_slice(&std::fs::read(
        parity.join("fixtures/parakeet-tdt-0.6b-v3/reference.json"),
    )?)?;
    let cases = document["cases"].as_array().context("fixture cases")?;
    let inputs: Value = serde_json::from_slice(&std::fs::read(wavs.join("inputs.json"))?)?;
    let profile = Profile::select(PROFILE, "transcribe")?;
    let speech_profile = manifest.profiles[PROFILE]
        .transcribe
        .as_ref()
        .context("speech profile")?;
    speech_profile.validate()?;
    let maximum = std::env::var("ANE_PARAKEET_MAX_DURATION_MS")
        .ok()
        .map(|s| s.parse::<u64>())
        .transpose()?
        .unwrap_or(speech_profile.max_duration_ms);
    ensure!(
        maximum <= speech_profile.max_duration_ms
            && cases.iter().any(
                |c| c["id"].as_str().is_some_and(|id| id.starts_with("ladder-"))
                    && c["duration_ms"] == maximum
            ),
        "maximum must be a sealed ladder duration within the profile envelope"
    );
    let smoke = std::env::var("ANE_PARAKEET_SMOKE").as_deref() == Ok("1");
    summary["mode"] = json!(if smoke {
        "smoke-not-certification"
    } else {
        "full"
    });
    summary["requested_max_duration_ms"] = json!(maximum);
    summary["kernel_revision"] = json!(synapse_core::ANE_DIRECT_TRANSCRIBE_KERNEL_REVISION);
    summary["decode_rule"] = json!(synapse_speech::DECODE_RULE_REVISION);
    summary["os_build"] = json!(String::from_utf8(
        std::process::Command::new("sw_vers")
            .arg("-buildVersion")
            .output()?
            .stdout
    )?
    .trim());
    summary["machine"] = json!(String::from_utf8(
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()?
            .stdout
    )?
    .trim());
    let tokenizer = synapse_speech::Tokenizer::from_pinned_bytes(&std::fs::read(
        snapshot.join("tokenizer.json"),
    )?)?;
    let mut model = load_model(&manifest, profile, &snapshot, &package)?;
    super::validate_load(&model.profile, synapse_core::DEFAULT_MAX_FRAME_BYTES)?;
    let frontend = synapse_speech::Frontend::new();
    let os = summary["os_build"].as_str().unwrap().to_owned();
    let mut resident = VecDeque::new();
    let mut wer = Vec::new();
    let mut max_live = 0;
    let mut largest_ladder = 0;
    let mut selected: Vec<_> = fixtures
        .iter()
        .zip(cases)
        .filter(|(f, c)| {
            !smoke
                || (!c["log_mel"].is_null() && f.duration_ms <= 3000 && f.split.is_some())
                || f.id == "silence-1s"
        })
        .collect();
    if smoke {
        selected.truncate(3);
    }
    ensure!(!selected.is_empty(), "no selected fixtures");
    // Short shapes first, then ascending ladder rungs, to make smoke and failure
    // receipts useful before the most expensive full-attention shape is built.
    selected.sort_by_key(|(f, _)| f.valid_encoder_frames);
    summary["cases"] = json!([]);
    for (fixture, case) in selected {
        summary["active_fixture"] = json!(fixture.id);
        save(summary)?;
        if fixture.duration_ms > maximum {
            ensure!(
                fixture.id.starts_with("ladder-"),
                "maximum does not cover the non-ladder corpus"
            );
            summary["cases"]
                .as_array_mut()
                .unwrap()
                .push(json!({"id": fixture.id,
                "status": "outside-selected-envelope", "engine_started": false}));
            continue;
        }
        let (mel, frames, valid) =
            fixture_features(case, &parity, &mirror, &wavs, &inputs, &frontend)?;
        let valid_encoder = super::validate_input(&model.profile, &mel, frames, valid)?;
        ensure!(
            valid_encoder as u64 == fixture.valid_encoder_frames,
            "valid encoder count: {}",
            fixture.id
        );
        let shape = crate::backend::rung(valid_encoder)?;
        if !resident.contains(&shape) {
            if resident.len() == 4 {
                model.evict(resident.pop_front().unwrap());
            }
            let mut progress = Progress::start();
            let admission = model.admit_progress(shape, &os, &mut || progress.completed())?;
            progress.finish(600.0)?;
            crate::backend::validate_inventory(&admission.inventory, 24)?;
            ensure!(
                admission.inventory.executables.len() == 24,
                "shape must have exactly 24 executables"
            );
            ensure!(
                progress.count == 24,
                "one setup completion is required per executable"
            );
            summary["shapes"].as_array_mut().unwrap().push(json!({"shape": shape,
                "executables": 24, "inventory": admission.inventory,
                "cache_hit_layers": admission.cache_hit_layers, "compiled_layers": admission.compiled_layers,
                "max_setup_gap_seconds": progress.maximum}));
            resident.push_back(shape);
        }
        let live: usize = model
            .resident
            .values()
            .map(|r| r.inventory.executables.len())
            .sum();
        max_live = max_live.max(live);
        ensure!(
            live <= 100 && model.resident.len() <= 4,
            "resident executable or shape budget exceeded"
        );
        let mut quiet_runs = Vec::new();
        let mut quiet_max_gap = 0.0f64;
        // Repeating the quiet mode distinguishes buffer reuse or nondeterminism
        // from an observation-only change. Readbacks copy completed work only.
        if !case["log_mel"].is_null() {
            for _ in 0..2 {
                let mut progress = Progress::start();
                let output = model.transcribe_progress(&mel, frames, valid, false, &mut || {
                    progress.completed()
                })?;
                progress.finish(120.0)?;
                quiet_max_gap = quiet_max_gap.max(progress.maximum);
                ensure!(
                    progress.count == 24 + valid_encoder / 125,
                    "quiet progress cadence"
                );
                ensure!(
                    output.encoder.is_empty() && output.decoded.trace.is_empty(),
                    "quiet request leaked observations"
                );
                let (encoder, projection) = model.transcribe_readbacks(output.frames)?;
                quiet_runs.push((output, encoder, projection));
            }
        }
        let started = Instant::now();
        let mut progress = Progress::start();
        let observed =
            model.transcribe_progress(&mel, frames, valid, true, &mut || progress.completed())?;
        progress.finish(120.0)?;
        ensure!(
            progress.count == 24 + valid_encoder / 125,
            "encoder/decoder progress cadence"
        );
        let elapsed = started.elapsed().as_secs_f64();
        let triples = observed.decoded.triples.clone();
        let parity_checked = !quiet_runs.is_empty();
        let mut parity_passes = true;
        summary["active_diagnostics"] = json!({"fixture": fixture.id});
        if let [(first, first_encoder, first_projection), (second, second_encoder, second_projection)] =
            quiet_runs.as_slice()
        {
            let (_, observed_projection) = model.transcribe_readbacks(observed.frames)?;
            let pairs = json!({
                "off_1_vs_off_2": compare_runs(first, first_encoder, first_projection, second, second_encoder, second_projection),
                "off_1_vs_on": compare_runs(first, first_encoder, first_projection, &observed, &observed.encoder, &observed_projection),
                "off_2_vs_on": compare_runs(second, second_encoder, second_projection, &observed, &observed.encoder, &observed_projection)});
            parity_passes = pairs
                .as_object()
                .unwrap()
                .values()
                .all(|pair| pair["triples_and_frames_equal"] == true);
            println!("PARAKEET_REPEAT {} {pairs}", fixture.id);
            summary["active_diagnostics"]["pairs"] = pairs;
        }
        save(summary)?;
        let cosine = (|| -> Result<Option<f64>> {
            if let Some(encoder) = &fixture.encoder {
                Ok(Some(speech::grade_encoder(
                    &f32le(&encoder.read(&parity, &mirror)?)?,
                    &observed.encoder,
                    valid_encoder,
                )?))
            } else {
                ensure!(
                    observed.encoder.len() == valid_encoder * 1024
                        && observed.encoder.iter().all(|x| x.is_finite()),
                    "malformed or nonfinite encoder output"
                );
                Ok(None)
            }
        })();
        let responses = crate::worker::transcribe_responses(
            "certify",
            observed,
            true,
            synapse_core::DEFAULT_MAX_FRAME_BYTES,
        )?;
        ensure!(
            responses.len() == 3
                && matches!(
                    responses[0].0,
                    synapse_core::WorkerResponse::EncoderOutput { .. }
                )
                && matches!(
                    responses[1].0,
                    synapse_core::WorkerResponse::DecodeTrace { .. }
                )
                && matches!(
                    responses[2].0,
                    synapse_core::WorkerResponse::Transcript { .. }
                ),
            "observation response order"
        );
        let trace = speech::decode_trace(responses[1].1.as_deref().unwrap())?;
        let signed: Vec<_> = triples.iter().map(|t| t.map(|x| x as i32)).collect();
        let exempt = speech::grade_trace(
            &speech::decode_trace(&fixture.trace.read(&parity, &mirror)?)?,
            &trace,
            &signed,
            valid_encoder as u64,
            speech_profile.tau_margin.unwrap(),
        );
        summary["active_diagnostics"]["encoder_gate"] = match &cosine {
            Ok(minimum) => json!({"status": "passed", "minimum_cosine": minimum}),
            Err(error) => json!({"status": "failed", "error": error.to_string()}),
        };
        summary["active_diagnostics"]["trace_gate"] = match &exempt {
            Ok(margin_exempt) => json!({"status": "passed", "margin_exempt": margin_exempt}),
            Err(error) => json!({"status": "failed", "error": error.to_string()}),
        };
        save(summary)?;
        ensure!(
            parity_passes,
            "off/off/on emitted triples or valid frames differ: {} (see active_diagnostics)",
            fixture.id
        );
        let cosine = cosine?;
        let exempt = exempt?;
        let text = tokenizer
            .transcript(&triples, fixture.duration_ms as f64)?
            .text;
        if let (Some(split), Some(truth)) = (&fixture.split, &fixture.ground_truth) {
            wer.push((split.clone(), truth.clone(), fixture.text.clone(), text));
        }
        if fixture.id.starts_with("ladder-") {
            largest_ladder = largest_ladder.max(fixture.duration_ms);
        }
        summary["max_live_executables"] = json!(max_live);
        summary["largest_passing_ladder_duration_ms"] = json!(largest_ladder);
        let diagnostics = summary["active_diagnostics"].clone();
        summary["cases"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id": fixture.id, "status": "passed",
            "valid_encoder_frames": valid_encoder, "shape": shape, "encoder_min_cosine": cosine,
            "trace_margin_exempt": exempt, "observe_on_off_bit_identical": parity_checked,
            "inference_seconds": elapsed, "progress_completions": progress.count,
            "max_inference_gap_seconds": progress.maximum.max(quiet_max_gap)}));
        summary["cases"].as_array_mut().unwrap().last_mut().unwrap()["diagnostics"] = diagnostics;
        eprintln!("parakeet certify: {} passed ({elapsed:.3}s)", fixture.id);
        save(summary)?;
    }
    let split_wer = speech::grade_wer(
        wer.iter()
            .map(|(s, t, r, a)| (s.as_str(), t.as_str(), r.as_str(), a.as_str())),
    )?;
    summary["split_wer"] = json!(split_wer);
    if !smoke {
        ensure!(
            split_wer.len() == 10,
            "full certification requires all ten WER splits"
        );
        ensure!(
            largest_ladder == maximum,
            "requested maximum ladder was not graded"
        );
        summary["certified_max_duration_ms"] = json!(maximum);
    }
    summary["active_fixture"] = Value::Null;
    Ok(())
}

fn tensor_comparison(left: &[f32], right: &[f32]) -> Value {
    let finite = left.iter().chain(right).all(|x| x.is_finite());
    let equal_length = left.len() == right.len();
    let maximum = (finite && equal_length).then(|| {
        left.iter()
            .zip(right)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0, f64::max)
    });
    json!({"max_abs_difference": maximum, "finite": finite, "equal_length": equal_length,
        "bit_identical": equal_length && left.iter().zip(right).all(|(a,b)| a.to_bits() == b.to_bits())})
}

#[test]
fn comparison_reports_truncation_frames_and_tensor_differences() {
    let output = |frames, triples| Output {
        frames,
        encoder: vec![],
        decoded: synapse_speech::DecodeOutput {
            triples,
            trace: vec![],
        },
    };
    let left = output(1, vec![[3, 0, 4]]);
    let right = output(2, vec![]);
    let comparison = compare_runs(&left, &[0.0, 1.0], &[1.0], &right, &[0.0, 1.5], &[2.0]);
    assert_eq!(comparison["valid_encoder_frames"], json!([1, 2]));
    assert_eq!(comparison["first_differing_triple_index"], 0);
    assert_eq!(comparison["left_triple"], json!([3, 0, 4]));
    assert!(comparison["right_triple"].is_null());
    assert_eq!(comparison["encoder"]["max_abs_difference"], 0.5);
    assert_eq!(comparison["projection"]["max_abs_difference"], 1.0);
    assert_eq!(comparison["triples_and_frames_equal"], false);
}

fn compare_runs(
    left: &Output,
    left_encoder: &[f32],
    left_projection: &[f32],
    right: &Output,
    right_encoder: &[f32],
    right_projection: &[f32],
) -> Value {
    let first = (0..left.decoded.triples.len().max(right.decoded.triples.len()))
        .find(|i| left.decoded.triples.get(*i) != right.decoded.triples.get(*i));
    json!({"valid_encoder_frames": [left.frames, right.frames],
        "triple_counts": [left.decoded.triples.len(), right.decoded.triples.len()],
        "first_differing_triple_index": first,
        "left_triple": first.and_then(|i| left.decoded.triples.get(i)),
        "right_triple": first.and_then(|i| right.decoded.triples.get(i)),
        "triples_and_frames_equal": left.frames == right.frames && first.is_none(),
        "encoder": tensor_comparison(left_encoder, right_encoder),
        "projection": tensor_comparison(left_projection, right_projection)})
}

fn load_model(
    manifest: &Manifest,
    profile: Profile,
    snapshot: &Path,
    package: &Path,
) -> Result<Model> {
    let digest = profile.numeric["converted_package_digest"]
        .as_str()
        .context("package digest")?
        .to_owned();
    if !package.exists() {
        let bytes = synapse_parity::convert::convert_profile_file(
            manifest,
            PROFILE,
            &snapshot.join("model.safetensors"),
        )?;
        ensure!(
            format!("sha256:{:x}", Sha256::digest(&bytes)) == digest,
            "converted package differs from pin"
        );
        std::fs::create_dir_all(package.parent().context("package parent")?)?;
        std::fs::write(package, bytes)?;
    }
    Model::load(profile, package, &digest)
}

fn compile_probes(
    summary: &mut Value,
    manifest: &Manifest,
    snapshot: &Path,
    package: &Path,
) -> Result<()> {
    let profile = Profile::select(PROFILE, "transcribe")?;
    let model = load_model(manifest, profile, snapshot, package)?;
    summary["mode"] = json!("compile-only-no-inference");
    summary["full_attention"] = Value::Null;
    summary["inference_calls"] = json!(0);
    summary["programs"] = json!([]);
    let mut failures = 0;
    let group = std::env::var("ANE_PARAKEET_PROBE_GROUP").unwrap_or_else(|_| "all".into());
    ensure!(
        ["all", "attention", "corrected"].contains(&group.as_str()),
        "unknown probe group"
    );
    summary["probe_group"] = json!(group);
    for name in super::probe_names()
        .into_iter()
        .filter(|name| match group.as_str() {
            "attention" => name.starts_with("attention"),
            "corrected" => [
                "attention-static-gather",
                "attention",
                "conformer-block",
                "full-layer-0",
            ]
            .contains(&name.as_str()),
            _ => true,
        })
    {
        let graph = super::probe_graph(&model, 128, &name)?;
        let (operations, inputs, outputs) = graph.diagnostic_shapes();
        let shapes = |shapes: Vec<Shape>| {
            shapes
                .into_iter()
                .map(|s| [s.batch, s.channels, s.height, s.width])
                .collect::<Vec<_>>()
        };
        let started = Instant::now();
        // Each executable is dropped and its autorelease pool drained before
        // compiling the next candidate. No inference request is ever created.
        let result = ane::autoreleasepool(|_| {
            graph
                .compile(ane::NSQualityOfService::UserInteractive)
                .map(drop)
        });
        let error = result.as_ref().err().map(ToString::to_string);
        if result.is_err() {
            failures += 1;
        }
        let row = json!({"program": name, "operation_count": operations,
            "input_shapes_nchw": shapes(inputs), "output_shapes_nchw": shapes(outputs),
            "status": if result.is_ok() { "passed" } else { "failed" },
            "error": error, "elapsed_seconds": started.elapsed().as_secs_f64()});
        println!("PARAKEET_COMPILE_PROBE {row}");
        summary["programs"].as_array_mut().unwrap().push(row);
        save(summary)?;
    }
    ensure!(
        failures == 0,
        "{failures} compile probe candidates failed (see per-program results)"
    );
    Ok(())
}

fn save(summary: &Value) -> Result<()> {
    let path = env_path("ANE_PARAKEET_REPORT")?;
    std::fs::create_dir_all(path.parent().context("report parent")?)?;
    std::fs::write(path, serde_json::to_vec_pretty(summary)?)?;
    Ok(())
}

#[test]
#[ignore = "requires hardware-owner opt-in, exclusive M5 ANE lane and pinned Parakeet assets"]
fn certify_parakeet() {
    assert_eq!(
        std::env::var("ANE_PARAKEET_CERTIFY").as_deref(),
        Ok("1"),
        "set ANE_PARAKEET_CERTIFY=1 explicitly"
    );
    let mut summary = json!({"status": "running", "shapes": [], "cases": [],
        "certified_max_duration_ms": null, "full_attention": true});
    let result = certify(&mut summary);
    summary["status"] = json!(if result.is_ok() { "passed" } else { "failed" });
    if let Err(error) = &result {
        summary["error"] = json!(format!("{error:#}"));
    }
    save(&summary).expect("write certification summary");
    result.unwrap();
}
