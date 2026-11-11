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
#[path = "parakeet_component_probe.rs"]
mod component_probe;
#[path = "parakeet_reporting.rs"]
mod reporting;

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
    started: Option<Instant>,
    encoder_seconds: Option<f64>,
    maximum: f64,
    count: usize,
}
impl Progress {
    fn start() -> Self {
        Self {
            previous: Some(Instant::now()),
            started: Some(Instant::now()),
            ..Self::default()
        }
    }
    fn completed(&mut self) -> Result<()> {
        self.maximum = self
            .maximum
            .max(self.previous.unwrap().elapsed().as_secs_f64());
        self.previous = Some(Instant::now());
        self.count += 1;
        if self.count == 24 {
            self.encoder_seconds = Some(self.started.unwrap().elapsed().as_secs_f64());
        }
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

fn inference_timing(mode: &str, progress: &Progress, seconds: f64) -> Value {
    json!({"mode": mode, "seconds": seconds, "encoder_seconds": progress.encoder_seconds,
        "decode_seconds": progress.encoder_seconds.map(|encoder| (seconds-encoder).max(0.0)),
        "progress_completions": progress.count, "max_progress_gap_seconds": progress.maximum})
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
    if std::env::var("ANE_PARAKEET_COMPONENT_PROBE").as_deref() == Ok("1") {
        return component_probe::run(summary, &manifest, &snapshot, &package, &parity);
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
    let timing = std::env::var("ANE_PARAKEET_TIMING").as_deref() == Ok("1");
    summary["mode"] = json!(if timing {
        "timing-not-certification"
    } else if smoke {
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
    let mut largest_processed_ladder = 0;
    let population = reporting::population(cases, maximum)?;
    let mut expected_wer = std::collections::BTreeMap::<String, usize>::new();
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
    if timing {
        let maximum_rung = std::env::var("ANE_PARAKEET_TIMING_MAX_RUNG")
            .ok()
            .map(|s| s.parse::<usize>())
            .transpose()?
            .unwrap_or(8192);
        ensure!(
            [128, 256, 512, 1024, 2048, 4096, 8192].contains(&maximum_rung),
            "unknown timing maximum rung"
        );
        summary["timing_max_rung"] = json!(maximum_rung);
        let mut ladder = std::collections::BTreeMap::new();
        let mut corpus = std::collections::BTreeMap::new();
        for pair in &selected {
            if pair.0.duration_ms > maximum {
                continue;
            }
            let shape = crate::backend::rung(pair.0.valid_encoder_frames as usize)?;
            if shape > maximum_rung {
                continue;
            }
            if pair.0.id.starts_with("ladder-") {
                let entry = ladder.entry(shape).or_insert(*pair);
                if pair.0.valid_encoder_frames > entry.0.valid_encoder_frames {
                    *entry = *pair;
                }
            } else if pair.0.split.is_some() && pair.1["log_mel"].is_null() {
                corpus.entry(shape).or_insert(*pair);
            }
        }
        selected = ladder.into_values().chain(corpus.into_values()).collect();
    }
    for (fixture, _) in &selected {
        if fixture.duration_ms <= maximum {
            if let Some(split) = &fixture.split {
                *expected_wer.entry(split.clone()).or_default() += 1;
            }
        }
    }
    ensure!(!selected.is_empty(), "no selected fixtures");
    // Short shapes first, then ascending ladder rungs, to make smoke and failure
    // receipts useful before the most expensive full-attention shape is built.
    selected.sort_by_key(|(f, _)| f.valid_encoder_frames);
    summary["cases"] = json!([]);
    let mut journal = reporting::open_case_journal()?;
    summary["case_journal"] = json!(env_path("ANE_PARAKEET_REPORT")?.with_extension("cases.jsonl"));
    let mut failed_shapes = std::collections::BTreeMap::<usize, String>::new();
    for (fixture, case) in selected {
        summary["active_fixture"] = json!(fixture.id);
        summary["active_diagnostics"] = json!({"fixture": fixture.id});
        eprintln!("parakeet certify: starting {}", fixture.id);
        if smoke
            || timing
            || summary["cases"]
                .as_array()
                .unwrap()
                .len()
                .is_multiple_of(25)
        {
            save(summary)?;
        }
        let case_started = Instant::now();
        let mut row = json!({"id": fixture.id, "split": fixture.split, "duration_ms": fixture.duration_ms,
            "shape": crate::backend::rung(fixture.valid_encoder_frames as usize).ok(),
            "stored_tensor": !case["log_mel"].is_null(), "mel_frames": case["mel_frames"],
            "phase": "prepare", "engine_started": false, "timing": {"admission_seconds": 0.0, "runs": []}});
        let mut gate_failures = Vec::<String>::new();
        let result = (|| -> Result<()> {
            if fixture.duration_ms > maximum {
                ensure!(
                    fixture.id.starts_with("ladder-"),
                    "maximum does not cover the non-ladder corpus"
                );
                if fixture.duration_ms > speech_profile.max_duration_ms {
                    let (mel, frames, valid) =
                        fixture_features(case, &parity, &mirror, &wavs, &inputs, &frontend)?;
                    let error = super::validate_input(&model.profile, &mel, frames, valid)
                        .err()
                        .context("above-limit input was admitted")?;
                    ensure!(
                        error.to_string() == "transcribe_duration_limit",
                        "wrong above-limit refusal: {error}"
                    );
                    row["status"] = json!("expected-refusal");
                    row["refusal"] = json!(error.to_string());
                } else {
                    row["status"] = json!("outside-selected-envelope");
                }
                return Ok(());
            }
            let prepared = Instant::now();
            let (mel, frames, valid) =
                fixture_features(case, &parity, &mirror, &wavs, &inputs, &frontend)?;
            row["timing"]["prepare_seconds"] = json!(prepared.elapsed().as_secs_f64());
            let valid_encoder = super::validate_input(&model.profile, &mel, frames, valid)?;
            ensure!(
                valid_encoder as u64 == fixture.valid_encoder_frames,
                "valid encoder count: {}",
                fixture.id
            );
            let shape = crate::backend::rung(valid_encoder)?;
            if let Some(error) = failed_shapes.get(&shape) {
                anyhow::bail!("shape unavailable after earlier engine failure: {error}");
            }
            row["phase"] = json!("setup");
            if !resident.contains(&shape) {
                summary["wer_details"] = reporting::wer_report(&wer, &expected_wer);
                summary["timing"] = reporting::timing_report(
                    summary["cases"].as_array().unwrap(),
                    &population,
                    summary["shapes"].as_array().unwrap(),
                );
                save(summary)?;
                if resident.len() == 4 {
                    model.evict(resident.pop_front().unwrap());
                }
                let mut progress = Progress::start();
                let setup_started = Instant::now();
                let admission = model.admit_progress(shape, &os, &mut || {
                    progress.completed()?;
                    eprintln!(
                        "parakeet setup: shape={shape} completed={}/24",
                        progress.count
                    );
                    Ok(())
                });
                let setup_seconds = setup_started.elapsed().as_secs_f64();
                row["timing"]["admission_seconds"] = json!(setup_seconds);
                if let Err(error) = progress.finish(600.0) {
                    gate_failures.push(error.to_string());
                }
                let setup = match &admission {
                    Ok(a) => {
                        json!({"shape": shape, "seconds": setup_seconds, "status": "loaded", "executables": 24,
                    "inventory": a.inventory, "cache_hit_layers": a.cache_hit_layers, "compiled_layers": a.compiled_layers,
                    "max_setup_gap_seconds": progress.maximum})
                    }
                    Err(error) => {
                        json!({"shape": shape, "seconds": setup_seconds, "status": "failed", "error": error.to_string(),
                    "max_setup_gap_seconds": progress.maximum})
                    }
                };
                summary["shapes"].as_array_mut().unwrap().push(setup);
                let admission = admission?;
                crate::backend::validate_inventory(&admission.inventory, 24)?;
                ensure!(
                    admission.inventory.executables.len() == 24,
                    "shape must have exactly 24 executables"
                );
                if progress.count != 24 {
                    gate_failures.push("one setup completion is required per executable".into());
                }
                resident.push_back(shape);
            }
            let live: usize = model
                .resident
                .values()
                .map(|r| r.inventory.executables.len())
                .sum();
            max_live = max_live.max(live);
            summary["max_live_executables"] = json!(max_live);
            ensure!(
                live <= 100 && model.resident.len() <= 4,
                "resident executable or shape budget exceeded"
            );
            let layer_reference = layer_reference(&fixture.id, valid_encoder, &model.profile)?;
            if layer_reference.is_some() {
                ensure!(
                    live < 100,
                    "no executable budget for isolated subsampling diagnostic"
                );
                capture_subsampling(&model, &mel, valid, shape, valid_encoder)?;
                summary["diagnostic_peak_live_executables"] = json!(live + 1);
            }
            let mut quiet_runs = Vec::new();
            let mut quiet_max_gap = 0.0f64;
            row["engine_started"] = json!(true);
            row["phase"] = json!("quiet-inference");
            // Repeating the quiet mode distinguishes buffer reuse or nondeterminism
            // from an observation-only change. Readbacks copy completed work only.
            if !case["log_mel"].is_null() {
                for _ in 0..2 {
                    let quiet_started = Instant::now();
                    let mut progress = Progress::start();
                    let output =
                        model.transcribe_progress(&mel, frames, valid, false, &mut || {
                            progress.completed()
                        })?;
                    let seconds = quiet_started.elapsed().as_secs_f64();
                    if let Err(error) = progress.finish(120.0) {
                        gate_failures.push(error.to_string());
                    }
                    row["timing"]["runs"]
                        .as_array_mut()
                        .unwrap()
                        .push(inference_timing("off", &progress, seconds));
                    quiet_max_gap = quiet_max_gap.max(progress.maximum);
                    if progress.count != 24 + valid_encoder / 125 {
                        gate_failures.push("quiet progress cadence".into());
                    }
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
            row["phase"] = json!("observed-inference");
            let observed = model
                .transcribe_progress(&mel, frames, valid, true, &mut || progress.completed())?;
            let elapsed = started.elapsed().as_secs_f64();
            row["phase"] = json!("grading");
            if let Err(error) = progress.finish(120.0) {
                gate_failures.push(error.to_string());
            }
            row["timing"]["runs"]
                .as_array_mut()
                .unwrap()
                .push(inference_timing("on", &progress, elapsed));
            if progress.count != 24 + valid_encoder / 125 {
                gate_failures.push("encoder/decoder progress cadence".into());
            }
            let triples = observed.decoded.triples.clone();
            if observed.frames != valid_encoder {
                gate_failures
                    .push("valid encoder frame count differs from the shared count".into());
            }
            row["decoder_work"] = json!(reporting::decoder_work(
                observed.decoded.trace.len(),
                triples.len()
            ));
            row["triples"] = json!(triples);
            row["valid_encoder_frames"] = json!(observed.frames);
            let text = tokenizer
                .transcript(&triples, fixture.duration_ms as f64)?
                .text;
            row["text"] = json!(text);
            row["reference_text"] = json!(fixture.text);
            row["ground_truth"] = json!(fixture.ground_truth);
            if let (Some(split), Some(truth)) = (&fixture.split, &fixture.ground_truth) {
                wer.push((split.clone(), truth.clone(), fixture.text.clone(), text));
            }
            let grading_started = Instant::now();
            let parity_checked = !quiet_runs.is_empty();
            let mut parity_passes = true;
            summary["active_diagnostics"] = json!({"fixture": fixture.id});
            if let Some((directory, index)) = layer_reference {
                summary["active_diagnostics"]["layer_comparisons"] =
                    compare_layer_reference(&directory, &index, valid_encoder)?;
            }
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
            if smoke || timing {
                save(summary)?;
            }
            let cosine = (|| -> Result<Option<f64>> {
                if let Some(encoder) = &fixture.encoder {
                    let reference = f32le(&encoder.read(&parity, &mirror)?)?;
                    let distribution = encoder_cosine_distribution(&reference, &observed.encoder)?;
                    println!(
                        "PARAKEET_ENCODER_DISTRIBUTION {} {distribution}",
                        fixture.id
                    );
                    summary["active_diagnostics"]["encoder_cosine_distribution"] = distribution;
                    Ok(Some(speech::grade_encoder(
                        &reference,
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
            let reference_trace = speech::decode_trace(&fixture.trace.read(&parity, &mirror)?)?;
            let exempt = speech::grade_trace(
                &reference_trace,
                &trace,
                &signed,
                valid_encoder as u64,
                speech_profile.tau_margin.unwrap(),
            );
            summary["active_diagnostics"]["trace_findings"] = reporting::trace_findings(
                &reference_trace,
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
            if smoke || timing {
                save(summary)?;
            }
            if !parity_passes {
                gate_failures.push("observe on/off triples or frame count mismatch".into());
            }
            if let Err(error) = &cosine {
                gate_failures.push(format!("encoder: {error}"));
            }
            if let Err(error) = &exempt {
                gate_failures.push(format!("trace: {error}"));
            }
            let passed = gate_failures.is_empty();
            if fixture.id.starts_with("ladder-") {
                largest_processed_ladder = largest_processed_ladder.max(fixture.duration_ms);
                if passed {
                    largest_ladder = largest_ladder.max(fixture.duration_ms);
                }
            }
            summary["max_live_executables"] = json!(max_live);
            summary["largest_passing_ladder_duration_ms"] = json!(largest_ladder);
            summary["largest_processed_ladder_duration_ms"] = json!(largest_processed_ladder);
            row["status"] = json!(if passed { "passed" } else { "grade-failed" });
            row["encoder_min_cosine"] =
                summary["active_diagnostics"]["encoder_cosine_distribution"]["minimum"].clone();
            row["trace_margin_exempt"] = json!(exempt.as_ref().ok());
            row["observe_on_off_checked"] = json!(parity_checked);
            row["observe_on_off_bit_identical"] = json!(parity_checked && parity_passes);
            row["inference_seconds"] = json!(elapsed);
            row["progress_completions"] = json!(progress.count);
            row["max_inference_gap_seconds"] = json!(progress.maximum.max(quiet_max_gap));
            row["timing"]["grading_seconds"] = json!(grading_started.elapsed().as_secs_f64());
            row["phase"] = json!("complete");
            eprintln!(
                "parakeet certify: {} {} ({elapsed:.3}s)",
                fixture.id, row["status"]
            );
            Ok(())
        })();
        if let Err(error) = result {
            if ["setup", "quiet-inference", "observed-inference"]
                .contains(&row["phase"].as_str().unwrap_or_default())
            {
                if let Some(shape) = row["shape"].as_u64() {
                    failed_shapes.insert(shape as usize, error.to_string());
                    model.evict(shape as usize);
                    resident.retain(|r| *r != shape as usize);
                }
            }
            row["status"] = json!("runtime-failed");
            row["error"] = json!(format!("{error:#}"));
        }
        row["diagnostics"] = summary["active_diagnostics"].clone();
        row["gate_failures"] = json!(gate_failures);
        row["timing"]["case_seconds"] = json!(case_started.elapsed().as_secs_f64());
        reporting::write_case_record(&mut journal, &row)?;
        summary["cases"].as_array_mut().unwrap().push(row);
        if smoke
            || timing
            || summary["cases"]
                .as_array()
                .unwrap()
                .len()
                .is_multiple_of(25)
        {
            summary["wer_details"] = reporting::wer_report(&wer, &expected_wer);
            save(summary)?;
        }
    }
    let split_wer = speech::grade_wer(
        wer.iter()
            .map(|(s, t, r, a)| (s.as_str(), t.as_str(), r.as_str(), a.as_str())),
    );
    let wer_details = reporting::wer_report(&wer, &expected_wer);
    let rates: serde_json::Map<String, Value> = wer_details
        .as_object()
        .unwrap()
        .iter()
        .map(|(split, row)| {
            (
                split.clone(),
                json!([row["reference_wer"], row["engine_wer"]]),
            )
        })
        .collect();
    summary["split_wer"] = Value::Object(rates);
    summary["wer_details"] = wer_details;
    summary["timing"] = reporting::timing_report(
        summary["cases"].as_array().unwrap(),
        &population,
        summary["shapes"].as_array().unwrap(),
    );
    let failed_cases = summary["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| {
            ["runtime-failed", "grade-failed"].contains(&row["status"].as_str().unwrap_or_default())
        })
        .count();
    let mut failures = Vec::new();
    if failed_cases > 0 {
        failures.push(format!(
            "{failed_cases} fixture(s) failed unchanged gates or runtime checks"
        ));
    }
    if let Err(error) = split_wer {
        failures.push(error.to_string());
    }
    if !smoke && !timing {
        if expected_wer.len() != 10
            || summary["wer_details"]
                .as_object()
                .unwrap()
                .values()
                .any(|r| r["complete"] != true)
        {
            failures
                .push("full certification requires every corpus case in all ten WER splits".into());
        }
        if largest_ladder != maximum {
            failures.push("requested maximum ladder did not pass every gate".into());
        }
    }
    summary["failed_case_count"] = json!(failed_cases);
    summary["gate_failures"] = json!(failures);
    if !smoke && !timing && failures.is_empty() {
        summary["certified_max_duration_ms"] = json!(maximum);
    }
    summary["active_fixture"] = Value::Null;
    save(summary)?;
    ensure!(
        failures.is_empty(),
        "collect-all run recorded failures; see complete per-case and WER reports"
    );
    Ok(())
}

fn encoder_cosine_distribution(reference: &[f32], actual: &[f32]) -> Result<Value> {
    ensure!(
        !reference.is_empty()
            && reference.len().is_multiple_of(1024)
            && reference.len() == actual.len(),
        "malformed encoder distribution tensors"
    );
    let values: Vec<_> = reference
        .as_chunks::<1024>()
        .0
        .iter()
        .zip(actual.as_chunks::<1024>().0.iter())
        .map(|(r, a)| {
            let (mut dot, mut rr, mut aa) = (0.0f64, 0.0f64, 0.0f64);
            for (r, a) in r.iter().zip(a) {
                let (r, a) = (f64::from(*r), f64::from(*a));
                dot += r * a;
                rr += r * r;
                aa += a * a;
            }
            dot / (rr * aa).sqrt()
        })
        .collect();
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite encoder distribution"
    );
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    Ok(
        json!({"frames": values.len(), "minimum": sorted[0], "maximum": sorted[sorted.len()-1],
        "mean": values.iter().sum::<f64>() / values.len() as f64, "p05": sorted[(sorted.len()-1)*5/100],
        "median": sorted[sorted.len()/2], "p95": sorted[(sorted.len()-1)*95/100],
        "below_0_999": values.iter().filter(|x| **x < 0.999).count(),
        "below_0_9999": values.iter().filter(|x| **x < 0.9999).count(),
        "indexed_frames": values.iter().enumerate().map(|(frame, cosine)| json!({"frame": frame,
            "distance_from_valid_end": values.len()-1-frame, "cosine": cosine})).collect::<Vec<_>>(), "per_frame": values}),
    )
}

fn layer_reference(
    fixture: &str,
    frames: usize,
    profile: &Profile,
) -> Result<Option<(PathBuf, Value)>> {
    let Some(path) = std::env::var_os("ANE_PARAKEET_LAYER_REFERENCE_INDEX") else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let index: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    if index["fixture"] != fixture {
        return Ok(None);
    }
    ensure!(
        index["schema"] == "parakeet-layer-reference-v1"
            && index["valid_encoder_frames"] == frames
            && index["checkpoint_sha256"] == profile.model["files"]["model.safetensors"],
        "layer reference binding mismatch"
    );
    env_path("ANE_PARAKEET_LAYER_DUMP_DIR")?;
    Ok(Some((
        path.parent()
            .context("layer reference directory")?
            .to_owned(),
        index,
    )))
}

fn capture_subsampling(
    model: &Model,
    mel: &[f32],
    valid: usize,
    shape: usize,
    frames: usize,
) -> Result<()> {
    ane::autoreleasepool(|_| {
        let graph = super::probe_graph(model, shape, "subsampling-complete")?;
        let executable = graph.compile(ane::NSQualityOfService::UserInteractive)?;
        let input = ane::TensorData::new(super::mel_shape(shape));
        let mask = ane::TensorData::new(crate::modernbert::shape(shape * 8, 1));
        let output = ane::TensorData::new(crate::modernbert::shape(shape, super::HIDDEN));
        let mut padded = vec![0.0; shape * 8 * 128];
        padded[..valid * 128].copy_from_slice(&mel[..valid * 128]);
        input.copy_from_f32(&padded);
        let mut valid_mask = vec![0.0; shape * 8];
        valid_mask[..valid].fill(1.0);
        mask.copy_from_f32(&valid_mask);
        executable.run(&[&input, &mask], &[&output])?;
        crate::backend::dump_transcribe_tensor("subsampling", shape, frames, &output)
    })
}

fn compare_layer_reference(directory: &Path, index: &Value, frames: usize) -> Result<Value> {
    let dump = env_path("ANE_PARAKEET_LAYER_DUMP_DIR")?;
    let mut report = serde_json::Map::new();
    for mode in ["reference", "rung", "converted"] {
        let mut rows = Vec::new();
        let mut first_drop = None;
        let names =
            std::iter::once("subsampling".to_owned()).chain((0..24).map(|n| format!("layer-{n}")));
        for name in names {
            let value = &index["artifacts"][format!("{mode}/{name}")];
            if value.is_null() {
                continue;
            }
            let expected = f32le(&artifact(value, directory, directory)?)?;
            let actual = f32le(&std::fs::read(
                dump.join(format!("{name}-frames-{frames}.f32le")),
            )?)?;
            let distribution = encoder_cosine_distribution(&expected, &actual)?;
            if distribution["minimum"].as_f64().unwrap() < 0.9999 && first_drop.is_none() {
                first_drop = Some(name.clone());
            }
            println!("PARAKEET_LAYER_COSINE {mode}/{name} {distribution}");
            rows.push(json!({"stage": name, "cosine": distribution}));
        }
        report.insert(
            mode.into(),
            json!({"first_below_0_9999": first_drop, "stages": rows}),
        );
    }
    Ok(Value::Object(report))
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
fn phase_timing_marks_completed_encoder_work_once() {
    let mut progress = Progress::start();
    for _ in 0..24 {
        progress.completed().unwrap();
    }
    let encoder = progress.encoder_seconds.unwrap();
    progress.completed().unwrap();
    assert_eq!(progress.encoder_seconds, Some(encoder));
    let split = inference_timing("on", &progress, encoder + 0.25);
    assert!((split["decode_seconds"].as_f64().unwrap() - 0.25).abs() < 1e-9);
    assert_eq!(split["progress_completions"], 25);
}

#[test]
fn encoder_distribution_reports_every_frame_without_hiding_gate_failures() {
    let mut reference = vec![0.0; 2048];
    let mut actual = vec![0.0; 2048];
    reference[0] = 1.0;
    reference[1025] = 1.0;
    actual[0] = 0.8;
    actual[1] = 0.6;
    actual[1025] = 1.0;
    let distribution = encoder_cosine_distribution(&reference, &actual).unwrap();
    assert_eq!(distribution["frames"], 2);
    assert_eq!(distribution["below_0_999"], 1);
    assert!((distribution["minimum"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    assert_eq!(distribution["maximum"], 1.0);
    assert_eq!(
        distribution["indexed_frames"][0]["distance_from_valid_end"],
        1
    );
    assert_eq!(distribution["indexed_frames"][1]["frame"], 1);
    assert_eq!(
        distribution["indexed_frames"][1]["distance_from_valid_end"],
        0
    );
    actual[0] = f32::NAN;
    assert!(encoder_cosine_distribution(&reference, &actual).is_err());
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
