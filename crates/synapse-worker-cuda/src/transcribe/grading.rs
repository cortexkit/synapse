//! Exhaustive hardware diagnostics; the parity evaluator remains the pass/fail gate.
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};
use synapse_parity::speech::{self, SpeechFixture, SpeechObservation, SpeechProfile, TraceStep};

fn distribution(mut values: Vec<f64>) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    values.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        let index = p * (values.len() - 1) as f64;
        let low = index.floor() as usize;
        let high = index.ceil() as usize;
        values[low] + (values[high] - values[low]) * index.fract()
    };
    json!({"frames": values.len(), "min": values[0], "max": values[values.len() - 1],
        "mean": values.iter().sum::<f64>() / values.len() as f64,
        "p01": quantile(0.01), "p05": quantile(0.05), "p50": quantile(0.5),
        "p95": quantile(0.95), "p99": quantile(0.99),
        "below_0_999": values.iter().filter(|&&v| v < 0.999).count()})
}

fn encoder_cosines(reference: &[f32], actual: &[f32], frames: usize) -> Result<Vec<f64>, String> {
    if frames == 0
        || reference.len() != frames * 1024
        || actual.len() != reference.len()
        || !reference.iter().chain(actual).all(|v| v.is_finite())
    {
        return Err("missing, malformed or nonfinite encoder observation".into());
    }
    let mut cosines = Vec::with_capacity(frames);
    for (index, (r, a)) in reference
        .chunks_exact(1024)
        .zip(actual.chunks_exact(1024))
        .enumerate()
    {
        // Match the evaluator's f64 accumulation order, but retain every frame
        // instead of returning when the first cosine falls below its threshold.
        let (mut dot, mut rr, mut aa) = (0., 0., 0.);
        for (&x, &y) in r.iter().zip(a) {
            let (x, y) = (f64::from(x), f64::from(y));
            dot += x * y;
            rr += x * x;
            aa += y * y;
        }
        let cosine = dot / (rr * aa).sqrt();
        if !cosine.is_finite() {
            return Err(format!("nonfinite encoder cosine at frame {index}"));
        }
        cosines.push(cosine);
    }
    Ok(cosines)
}

#[derive(Default)]
struct WerCounts {
    words: usize,
    reference: usize,
    engine: usize,
    fixtures: usize,
}

impl WerCounts {
    fn add(&mut self, truth: &str, reference: &str, engine: &str) {
        let truth = speech::normalize_words(truth);
        self.words += truth.len();
        self.reference += speech::word_edits(&truth, &speech::normalize_words(reference));
        self.engine += speech::word_edits(&truth, &speech::normalize_words(engine));
        self.fixtures += 1;
    }

    fn report(&self) -> Value {
        if self.words == 0 {
            return json!({"fixtures": self.fixtures, "error": "no ground-truth words"});
        }
        let reference = self.reference as f64 / self.words as f64;
        let engine = self.engine as f64 / self.words as f64;
        json!({"fixtures": self.fixtures, "ground_truth_words": self.words,
            "reference_edits": self.reference, "engine_edits": self.engine,
            "reference_wer": reference, "engine_wer": engine,
            "delta_percentage_points": 100. * (engine - reference),
            "passed": engine - reference <= 0.005})
    }
}

pub(super) fn collect(
    profile: &SpeechProfile,
    fixtures: &[SpeechFixture],
    observations: &BTreeMap<String, SpeechObservation>,
    parity: &Path,
    mirror: &Path,
) -> Value {
    let mut errors = Vec::new();
    if let Err(error) = profile.validate() {
        errors.push(json!({"stage": "profile", "error": error.to_string()}));
    }
    let tau = profile.tau_margin.unwrap_or(f64::NAN);
    let mut trace_failures = Vec::new();
    let mut margin_exempt = Vec::new();
    let mut trace_equal = 0;
    let mut encoder_rows = Vec::new();
    let mut all_cosines = Vec::new();
    let mut wer: BTreeMap<String, WerCounts> = BTreeMap::new();
    let mut expected_refusals = Vec::new();
    for fixture in fixtures {
        let id = &fixture.id;
        let Some(actual) = observations.get(id) else {
            errors.push(
                json!({"fixture": id, "stage": "observation", "error": "missing observation"}),
            );
            continue;
        };
        match speech::grade_applicability(
            profile,
            "duration_limit",
            fixture.duration_ms,
            fixture.sample_count,
            actual.valid_encoder_frames,
            actual.refusal.as_ref(),
            actual.engine_started,
        ) {
            Ok(false) => {
                expected_refusals.push(id.clone());
                continue;
            }
            Ok(true) => {}
            Err(error) => errors
                .push(json!({"fixture": id, "stage": "applicability", "error": error.to_string()})),
        }
        if let (Some(split), Some(truth)) = (&fixture.split, &fixture.ground_truth) {
            wer.entry(split.clone())
                .or_default()
                .add(truth, &fixture.text, &actual.text);
        } else if fixture.split.is_some() || fixture.ground_truth.is_some() {
            errors.push(json!({"fixture": id, "stage": "wer", "error": "split and ground truth must both be present"}));
        }
        if actual.encoder_output.as_ref().is_none_or(|values| {
            values.len() != fixture.valid_encoder_frames as usize * 1024
                || values.iter().any(|v| !v.is_finite())
        }) {
            errors.push(json!({"fixture": id, "stage": "encoder_observation", "error": "missing, malformed or nonfinite encoder observation"}));
        }
        if let Some(artifact) = &fixture.encoder {
            let result = artifact
                .read(parity, mirror)
                .map_err(|e| e.to_string())
                .and_then(|raw| {
                    if raw.len() % 4 != 0 {
                        return Err("encoder artifact is not f32le".into());
                    }
                    let reference: Vec<f32> = raw
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    let actual = actual.encoder_output.as_deref().unwrap_or_default();
                    let cosines =
                        encoder_cosines(&reference, actual, fixture.valid_encoder_frames as usize)?;
                    let gate = speech::grade_encoder(
                        &reference,
                        actual,
                        fixture.valid_encoder_frames as usize,
                    );
                    let row = json!({"fixture": id, "distribution": distribution(cosines.clone()),
                    "passed": gate.is_ok(), "error": gate.err().map(|e| e.to_string())});
                    all_cosines.extend(cosines);
                    Ok(row)
                });
            match result {
                Ok(row) => encoder_rows.push(row),
                Err(error) => {
                    errors.push(json!({"fixture": id, "stage": "encoder", "error": error}))
                }
            }
        }
        let reference = fixture
            .trace
            .read(parity, mirror)
            .and_then(|raw| speech::decode_trace(&raw));
        match reference {
            Ok(reference) => {
                let gate = speech::grade_trace(
                    &reference,
                    &actual.trace,
                    &actual.triples,
                    fixture.valid_encoder_frames,
                    tau,
                );
                let difference = first_trace_difference(&reference, &actual.trace, tau);
                match gate {
                    Ok(false) => trace_equal += 1,
                    Ok(true) => {
                        let mut detail =
                            difference.unwrap_or_else(|| json!({"reason": "trace_validation"}));
                        detail["fixture"] = json!(id);
                        margin_exempt.push(detail);
                    }
                    Err(error) => {
                        let mut detail =
                            difference.unwrap_or_else(|| json!({"reason": "trace_validation"}));
                        detail["fixture"] = json!(id);
                        detail["error"] = json!(error.to_string());
                        detail["margin_exempt"] = json!(false);
                        trace_failures.push(detail);
                    }
                }
            }
            Err(error) => errors.push(
                json!({"fixture": id, "stage": "reference_trace", "error": error.to_string()}),
            ),
        }
    }
    let split_wer: BTreeMap<_, _> = wer
        .into_iter()
        .map(|(split, counts)| (split, counts.report()))
        .collect();
    json!({"fixture_count": fixtures.len(), "observation_count": observations.len(),
        "thresholds": {"tau_margin": profile.tau_margin, "minimum_encoder_cosine": 0.999,
            "maximum_split_wer_delta_percentage_points": 0.5},
        "trace_equal": trace_equal, "trace_failures": trace_failures, "margin_exempt": margin_exempt,
        "encoder": {"distribution": distribution(all_cosines), "fixtures": encoder_rows},
        "split_wer": split_wer, "expected_refusals": expected_refusals, "errors": errors})
}

fn head(top1: i32, top2: i32, logit1: f32, logit2: f32) -> Value {
    json!({"top1": top1, "top2": top2, "logit1": logit1, "logit2": logit2,
        "margin": logit1 - logit2})
}

fn step_detail(step: &TraceStep) -> Value {
    json!({"frame_index": step.frame_index,
        "token": head(step.token_top1, step.token_top2, step.token_logit1, step.token_logit2),
        "duration": head(step.duration_top1, step.duration_top2, step.duration_logit1, step.duration_logit2)})
}

fn first_trace_difference(
    reference: &[TraceStep],
    engine: &[TraceStep],
    tau: f64,
) -> Option<Value> {
    for index in 0..reference.len().max(engine.len()) {
        let (Some(r), Some(e)) = (reference.get(index), engine.get(index)) else {
            return Some(
                json!({"step": index, "reason": "trace_length", "margin_exempt": false,
                "reference": reference.get(index).map(step_detail),
                "engine": engine.get(index).map(step_detail)}),
            );
        };
        let token_differs = r.token_top1 != e.token_top1;
        let duration_differs = r.duration_top1 != e.duration_top1;
        if r.frame_index == e.frame_index && !token_differs && !duration_differs {
            continue;
        }
        let mut heads = Vec::new();
        if token_differs {
            heads.push("token");
        }
        if duration_differs {
            heads.push("duration");
        }
        let exempt = r.frame_index == e.frame_index
            && (!token_differs || f64::from(r.token_logit1 - r.token_logit2) < tau)
            && (!duration_differs || f64::from(r.duration_logit1 - r.duration_logit2) < tau);
        return Some(json!({"step": index,
            "reason": if r.frame_index == e.frame_index { "argmax" } else { "frame_index" },
            "heads": heads, "margin_exempt": exempt,
            "reference": step_detail(r), "engine": step_detail(e)}));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step() -> TraceStep {
        TraceStep {
            frame_index: 417,
            token_top1: 7990,
            token_top2: 497,
            duration_top1: 4,
            duration_top2: 3,
            token_logit1: 10.,
            token_logit2: 1.,
            duration_logit1: 11.35513,
            duration_logit2: 11.35067,
            emitted: true,
            cap_hit: false,
        }
    }

    #[test]
    fn transcribe_trace_diagnostic_names_head_pairs_and_frozen_margin_failure() {
        let reference = step();
        let mut engine = reference.clone();
        engine.duration_top1 = 3;
        engine.duration_top2 = 4;
        let difference = first_trace_difference(&[reference], &[engine], 0.001).unwrap();
        assert_eq!(difference["step"], 0);
        assert_eq!(difference["heads"], json!(["duration"]));
        assert_eq!(difference["reference"]["frame_index"], 417);
        assert_eq!(difference["reference"]["duration"]["top1"], 4);
        assert_eq!(difference["engine"]["duration"]["top1"], 3);
        assert_eq!(difference["engine"]["duration"]["top2"], 4);
        assert_eq!(difference["margin_exempt"], false);
        let margin = difference["reference"]["duration"]["margin"]
            .as_f64()
            .unwrap();
        assert!((margin - 0.00446).abs() < 0.000001);
    }

    #[test]
    fn transcribe_trace_diagnostic_keeps_frame_and_length_failures_nonexempt() {
        let reference = step();
        let mut engine = reference.clone();
        engine.frame_index += 1;
        let difference =
            first_trace_difference(std::slice::from_ref(&reference), &[engine], 100.).unwrap();
        assert_eq!(difference["reason"], "frame_index");
        assert_eq!(difference["margin_exempt"], false);
        let difference = first_trace_difference(&[reference], &[], 100.).unwrap();
        assert_eq!(difference["reason"], "trace_length");
        assert_eq!(difference["margin_exempt"], false);
        assert!(difference["engine"].is_null());
    }

    #[test]
    fn transcribe_diagnostic_wer_uses_total_words_and_edits() {
        let mut counts = WerCounts::default();
        counts.add("A b!", "a b", "a");
        counts.add("c d e f g h", "c d e f g h", "c d e f g h");
        let report = counts.report();
        assert_eq!(report["ground_truth_words"], 8);
        assert_eq!(report["reference_edits"], 0);
        assert_eq!(report["engine_edits"], 1);
        assert_eq!(report["engine_wer"], 0.125);
        assert_eq!(report["delta_percentage_points"], 12.5);
        assert_eq!(report["passed"], false);
    }

    #[test]
    fn transcribe_diagnostics_retain_all_failures_wer_and_encoder_distribution() {
        use sha2::{Digest, Sha256};
        use synapse_core::{encode_decode_trace_frame, encode_f32_frame, DecodeTraceRecord};
        use synapse_parity::speech::Artifact;

        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../target/scratch/cuda-grading-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let artifact = |name: &str, raw: Vec<u8>| {
            std::fs::write(directory.join(name), &raw).unwrap();
            Artifact {
                path: Some(name.into()),
                cache_path: None,
                sha256: format!("{:x}", Sha256::digest(&raw)),
                bytes: raw.len() as u64,
            }
        };
        let mut reference = step();
        reference.frame_index = 0;
        let trace = artifact(
            "trace.bin",
            encode_decode_trace_frame(&[DecodeTraceRecord {
                frame_index: 0,
                token_top1: 7990,
                token_top2: 497,
                duration_top1: 4,
                duration_top2: 3,
                token_logit1: 10.,
                token_logit2: 1.,
                duration_logit1: 11.35513,
                duration_logit2: 11.35067,
                flags: 1,
            }]),
        );
        let mut basis = vec![0.; 1024];
        basis[0] = 1.;
        let encoder = artifact("encoder.bin", encode_f32_frame(&basis));
        let profile = SpeechProfile {
            max_duration_ms: 600000,
            encoder_rungs: vec![128, 256, 512, 1024, 2048, 4096, 8192],
            tau_feat: Some(0.001),
            tau_resample: Some(0.001),
            tau_margin: Some(0.001),
            confirmations: json!({"transformers_version": "5.16.1"}),
            self_check: trace.clone(),
        };
        let fixtures: Vec<_> = [("case-a", "de"), ("case-b", "fr")]
            .into_iter()
            .map(|(id, split)| SpeechFixture {
                id: id.into(),
                sample_count: 160,
                duration_ms: 10,
                valid_encoder_frames: 1,
                trace: trace.clone(),
                encoder: Some(encoder.clone()),
                text: "a b".into(),
                split: Some(split.into()),
                ground_truth: Some("a b".into()),
            })
            .collect();
        let mut engine = reference.clone();
        engine.duration_top1 = 3;
        engine.duration_top2 = 4;
        let mut observations: BTreeMap<_, _> = fixtures
            .iter()
            .map(|fixture| {
                (
                    fixture.id.clone(),
                    SpeechObservation {
                        valid_encoder_frames: Some(1),
                        engine_started: true,
                        encoder_output: Some(basis.clone()),
                        trace: vec![engine.clone()],
                        triples: vec![[7990, 0, 3]],
                        text: "a b".into(),
                        ..Default::default()
                    },
                )
            })
            .collect();
        observations.get_mut("case-a").unwrap().text = "a".into();
        let second = observations
            .get_mut("case-b")
            .unwrap()
            .encoder_output
            .as_mut()
            .unwrap();
        second[0] = 0.;
        second[1] = 1.;

        let report = collect(&profile, &fixtures, &observations, &directory, &directory);
        assert_eq!(report["trace_failures"].as_array().unwrap().len(), 2);
        assert_eq!(report["trace_failures"][0]["fixture"], "case-a");
        assert_eq!(report["trace_failures"][1]["fixture"], "case-b");
        assert_eq!(report["split_wer"]["de"]["engine_wer"], 0.5);
        assert_eq!(report["split_wer"]["fr"]["engine_wer"], 0.);
        assert_eq!(report["encoder"]["distribution"]["frames"], 2);
        assert_eq!(report["encoder"]["distribution"]["min"], 0.);
        assert_eq!(report["encoder"]["distribution"]["max"], 1.);
        assert_eq!(report["encoder"]["distribution"]["p50"], 0.5);
        assert_eq!(report["encoder"]["distribution"]["below_0_999"], 1);
        assert_eq!(report["encoder"]["fixtures"][1]["passed"], false);
        assert!(report["errors"].as_array().unwrap().is_empty());
        // Diagnostic collection must not turn a canonical failure into a pass.
        assert!(speech::evaluate_transcribe(
            &profile,
            &fixtures,
            &observations,
            &directory,
            &directory,
            "duration_limit"
        )
        .is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
