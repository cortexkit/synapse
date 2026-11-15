//! Owner-only hardware execution of the compiled first-load speech references.
use super::*;

pub(super) fn run(
    summary: &mut Value,
    manifest: &Manifest,
    snapshot: &Path,
    package: &Path,
) -> Result<()> {
    let profile = Profile::select(PROFILE, "transcribe")?;
    let speech_profile = manifest.profiles[PROFILE]
        .transcribe
        .as_ref()
        .context("speech profile")?;
    speech_profile.validate()?;
    summary["mode"] = json!("first-load-self-check");
    summary["precision_class"] = json!(speech_profile.precision_class);
    summary["cases"] = json!([]);
    summary["certified_max_duration_ms"] = Value::Null;
    let cases = synapse_certify::self_check::speech::load(PROFILE)?;
    let tokenizer = synapse_speech::Tokenizer::from_pinned_bytes(&std::fs::read(
        snapshot.join("tokenizer.json"),
    )?)?;
    let frontend = synapse_speech::Frontend::new();
    let mut model = load_model(manifest, profile, snapshot, package)?;
    let os = String::from_utf8(
        std::process::Command::new("sw_vers")
            .arg("-buildVersion")
            .output()?
            .stdout,
    )?;
    let mut resident = Vec::new();
    let mut failures = 0;
    for case in cases {
        let started = Instant::now();
        let mut row = json!({"id": case.id, "reference_text": case.text});
        let result = (|| -> Result<()> {
            let features = frontend.process_wav(&case.wav)?;
            let frames = super::super::validate_input(
                &model.profile,
                &features.values,
                features.mel_frames,
                features.valid_mel_frames,
            )?;
            ensure!(
                frames == case.valid_encoder_frames,
                "self-check frame count mismatch"
            );
            let shape = crate::backend::rung(frames)?;
            if !resident.contains(&shape) {
                let mut progress = Progress::start();
                model.admit_progress(shape, os.trim(), &mut || progress.completed())?;
                progress.finish(180.0)?;
                ensure!(progress.count == 24, "self-check setup progress cadence");
                resident.push(shape);
            }
            let mut progress = Progress::start();
            let output = model.transcribe_progress(
                &features.values,
                features.mel_frames,
                features.valid_mel_frames,
                false,
                &mut || progress.completed(),
            )?;
            progress.finish(120.0)?;
            ensure!(
                progress.count == 24 + frames / 125,
                "self-check inference progress cadence"
            );
            let triples: Vec<_> = output
                .decoded
                .triples
                .iter()
                .map(|t| t.map(|v| v as i32))
                .collect();
            let transcript =
                tokenizer.transcript(&output.decoded.triples, case.duration_ms as f64)?;
            row["text"] = json!(transcript.text);
            row["reference_triples_equal"] = json!(triples == case.triples);
            row["triples"] = json!(triples);
            let responses = crate::worker::transcribe_responses(
                "self-check",
                output,
                false,
                synapse_core::DEFAULT_MAX_FRAME_BYTES,
            )?;
            ensure!(
                responses.len() == 1
                    && matches!(
                        responses[0].0,
                        synapse_core::WorkerResponse::Transcript { .. }
                    ),
                "self-check protocol response"
            );
            synapse_certify::self_check::speech::compare(&case, &triples, &transcript.text)?;
            Ok(())
        })();
        row["seconds"] = json!(started.elapsed().as_secs_f64());
        match result {
            Ok(()) => row["status"] = json!("passed"),
            Err(error) => {
                failures += 1;
                row["status"] = json!("failed");
                row["error"] = json!(format!("{error:#}"));
            }
        }
        summary["cases"].as_array_mut().unwrap().push(row);
        save(summary)?;
    }
    ensure!(
        failures == 0,
        "{failures} first-load speech self-check(s) failed"
    );
    Ok(())
}
