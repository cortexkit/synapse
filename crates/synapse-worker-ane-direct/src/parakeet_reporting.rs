//! Collect-all diagnostics retain failed grades without changing their rules.
use super::*;
use std::collections::BTreeMap;

pub(super) fn trace_findings(
    reference: &[speech::TraceStep],
    actual: &[speech::TraceStep],
    triples: &[speech::Triple],
    frames: u64,
    tau: f64,
) -> Value {
    let grade = speech::grade_trace(reference, actual, triples, frames, tau);
    let implied = speech::implied_triples(actual, frames);
    let consistent = implied.as_ref().is_ok_and(|expected| expected == triples);
    let mut examples = Vec::new();
    let mut differences = 0;
    for step in 0..reference.len().max(actual.len()) {
        let (r, a) = (reference.get(step), actual.get(step));
        let kind = match (r, a) {
            (Some(r), Some(a)) if r.frame_index != a.frame_index => "frame",
            (Some(r), Some(a))
                if r.token_top1 != a.token_top1 || r.duration_top1 != a.duration_top1 =>
            {
                "argmax"
            }
            (Some(_), Some(_)) => continue,
            _ => "length",
        };
        differences += 1;
        if examples.len() < 16 {
            examples.push(
                json!({"step": step, "kind": kind, "reference": r, "actual": a,
                "reference_token_margin": r.map(|s| s.token_logit1 - s.token_logit2),
                "reference_duration_margin": r.map(|s| s.duration_logit1 - s.duration_logit2)}),
            );
        }
    }
    json!({"status": if grade.is_ok() { "passed" } else { "failed" },
        "margin_exempt": grade.as_ref().ok(), "error": grade.as_ref().err().map(ToString::to_string),
        "reference_steps": reference.len(), "actual_steps": actual.len(), "tau_margin": tau,
        "whole_trace_triples_consistent": consistent,
        "whole_trace_error": implied.as_ref().err().map(ToString::to_string),
        "first_difference": examples.first(), "different_steps": differences,
        "difference_examples": examples, "example_limit": 16})
}

pub(super) fn wer_report(
    rows: &[(String, String, String, String)],
    expected_cases: &BTreeMap<String, usize>,
) -> Value {
    let mut groups: BTreeMap<&str, Vec<(&str, &str, &str)>> = BTreeMap::new();
    for (split, truth, reference, actual) in rows {
        groups
            .entry(split)
            .or_default()
            .push((truth, reference, actual));
    }
    let mut report = serde_json::Map::new();
    for (split, expected) in expected_cases {
        let rows = groups
            .get(split.as_str())
            .map(Vec::as_slice)
            .unwrap_or_default();
        let (mut words, mut reference_edits, mut actual_edits) = (0, 0, 0);
        for (truth, reference, actual) in rows {
            let truth = speech::normalize_words(truth);
            words += truth.len();
            reference_edits += speech::word_edits(&truth, &speech::normalize_words(reference));
            actual_edits += speech::word_edits(&truth, &speech::normalize_words(actual));
        }
        let rates = (words > 0).then(|| {
            [
                reference_edits as f64 / words as f64,
                actual_edits as f64 / words as f64,
            ]
        });
        let gate = speech::grade_wer(rows.iter().map(|(t, r, a)| (split.as_str(), *t, *r, *a)));
        let complete = rows.len() == *expected;
        report.insert(split.clone(), json!({"expected_cases": expected, "observed_cases": rows.len(),
            "complete": complete, "ground_truth_words": words, "reference_edits": reference_edits,
            "engine_edits": actual_edits, "reference_wer": rates.map(|r| r[0]), "engine_wer": rates.map(|r| r[1]),
            "delta_percentage_points": rates.map(|r| (r[1]-r[0])*100.0),
            "status": if !complete { "incomplete" } else if gate.is_err() { "failed" } else { "passed" },
            "error": gate.err().map(|e| e.to_string())}));
    }
    Value::Object(report)
}

pub(super) fn decoder_work(steps: usize, emissions: usize) -> f64 {
    // A joint head evaluates 8198 by 640 weights. Each prediction evaluates two
    // LSTM layers (input and recurrent 2560-by-640 matrices) and a 640 projection.
    steps as f64 * (8198 * 640) as f64
        + (emissions + 1) as f64 * (4 * 2560 * 640 + 640 * 640) as f64
}

pub(super) fn population(cases: &[Value], maximum: u64) -> Result<Vec<Value>> {
    cases.iter().filter(|c| c["duration_ms"].as_u64().is_some_and(|ms| ms <= maximum)).map(|c| {
        let frames = c["valid_encoder_frames"].as_u64().context("population encoder frames")? as usize;
        let steps = c["trace"]["bytes"].as_u64().context("population trace bytes")? as usize / 40;
        let tokens = c["triples"].as_array().context("population triples")?.len();
        Ok(json!({"id": c["id"], "rung": crate::backend::rung(frames)?, "stored": !c["log_mel"].is_null(),
            "mel_frames": c["mel_frames"], "decoder_work": decoder_work(steps, tokens)}))
    }).collect()
}

pub(super) fn timing_report(cases: &[Value], population: &[Value], admissions: &[Value]) -> Value {
    let mut report = serde_json::Map::new();
    let mut estimated_total = 0.0;
    let mut covered_cases = 0;
    let mut available = true;
    let mut prepare = [[0.0f64; 2]; 2];
    let mut grading = [0.0f64; 2];
    for case in cases {
        let timing = &case["timing"];
        let stored = usize::from(case["stored_tensor"].as_bool().unwrap_or(false));
        if let (Some(seconds), Some(mel)) = (
            timing["prepare_seconds"].as_f64(),
            case["mel_frames"].as_f64(),
        ) {
            prepare[stored][0] += seconds;
            prepare[stored][1] += mel;
        }
        if let (Some(seconds), Some(mel)) = (
            timing["grading_seconds"].as_f64(),
            case["mel_frames"].as_f64(),
        ) {
            grading[0] += seconds;
            grading[1] += mel;
        }
    }
    for rung in [128, 256, 512, 1024, 2048, 4096, 8192] {
        let target: Vec<_> = population.iter().filter(|p| p["rung"] == rung).collect();
        if target.is_empty() {
            continue;
        }
        let samples: Vec<_> = cases.iter().filter(|c| c["shape"] == rung).collect();
        let mut calls = 0;
        let (mut encoder, mut decode, mut work, mut max_gap) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut timings = Vec::new();
        for case in &samples {
            if let Some(runs) = case["timing"]["runs"].as_array() {
                for run in runs {
                    if let (Some(e), Some(d), Some(w)) = (
                        run["encoder_seconds"].as_f64(),
                        run["decode_seconds"].as_f64(),
                        case["decoder_work"].as_f64(),
                    ) {
                        calls += 1;
                        encoder += e;
                        decode += d;
                        work += w;
                        timings.push(run["seconds"].as_f64().unwrap_or(e + d));
                        max_gap =
                            max_gap.max(run["max_progress_gap_seconds"].as_f64().unwrap_or(0.0));
                    }
                }
            }
        }
        let full_calls: usize = target
            .iter()
            .map(|p| if p["stored"] == true { 3 } else { 1 })
            .sum();
        let full_work: f64 = target
            .iter()
            .map(|p| {
                p["decoder_work"].as_f64().unwrap() * if p["stored"] == true { 3.0 } else { 1.0 }
            })
            .sum();
        let mut estimated_prepare = 0.0;
        let mut prep_available = true;
        for p in &target {
            let row = prepare[usize::from(p["stored"] == true)];
            if row[1] == 0.0 {
                prep_available = false;
            } else {
                estimated_prepare += row[0] / row[1] * p["mel_frames"].as_f64().unwrap();
            }
        }
        let full_mel: f64 = target
            .iter()
            .map(|p| p["mel_frames"].as_f64().unwrap())
            .sum();
        let estimated_grading = if grading[1] > 0.0 {
            grading[0] / grading[1] * full_mel
        } else {
            0.0
        };
        let setup: Vec<_> = admissions.iter().filter(|a| a["shape"] == rung).collect();
        let setup_seconds: f64 = setup.iter().filter_map(|a| a["seconds"].as_f64()).sum();
        let estimate = (calls > 0 && work > 0.0 && prep_available).then(|| {
            encoder / calls as f64 * full_calls as f64
                + decode / work * full_work
                + estimated_prepare
                + estimated_grading
                + setup_seconds
        });
        if let Some(seconds) = estimate {
            estimated_total += seconds;
            covered_cases += target.len();
        } else {
            available = false;
        }
        report.insert(rung.to_string(), json!({"population_cases": target.len(), "population_engine_calls": full_calls,
            "measured_cases": samples.len(), "measured_engine_calls": calls, "measured_run_seconds": timings,
            "encoder_seconds_total": encoder, "decode_seconds_total": decode, "max_progress_gap_seconds": max_gap,
            "admissions": setup, "observed_admission_seconds": setup_seconds,
            "estimated_prepare_seconds": if prep_available { Some(estimated_prepare) } else { None },
            "estimated_grading_seconds": estimated_grading, "estimated_full_seconds_including_observed_setup": estimate}));
    }
    json!({"by_rung": report, "complete": available,
        "measured_rung_population_cases": covered_cases, "population_cases": population.len(),
        "estimated_measured_rung_seconds": estimated_total,
        "estimated_full_seconds_including_observed_setup": available.then_some(estimated_total),
        "suggested_window_seconds_with_50pct_margin": available.then_some(estimated_total*1.5),
        "assumptions": ["Encoder time is weighted by padded rung and number of observe/parity calls.",
            "Decoder time is weighted by sealed reference joint/predictor matrix work; actual traces may differ.",
            "Preparation and grading scale by mel frames; setup uses the measured cache-hit/cold-compile mix.",
            "The suggested extra margin covers report checkpointing, cache-load variation and thermal effects; it is not a deadline guarantee.",
            "Only successful owner-run admissions establish cache state; eviction or daemon cache clearing can change it."]})
}

#[test]
fn wer_reporting_retains_all_splits_even_when_the_authoritative_gate_fails() {
    let rows = vec![
        (
            "a".into(),
            "one two".into(),
            "one two".into(),
            "one wrong".into(),
        ),
        ("b".into(), "good".into(), "good".into(), "good".into()),
    ];
    let expected = BTreeMap::from([("a".into(), 1), ("b".into(), 1)]);
    let report = wer_report(&rows, &expected);
    assert_eq!(report.as_object().unwrap().len(), 2);
    assert_eq!(report["a"]["engine_wer"], 0.5);
    assert_eq!(report["a"]["status"], "failed");
    assert_eq!(report["b"]["status"], "passed");
    assert!(speech::grade_wer(rows.iter().map(|(s, t, r, a)| (
        s.as_str(),
        t.as_str(),
        r.as_str(),
        a.as_str()
    )))
    .is_err());
    let incomplete = wer_report(&rows[..1], &expected);
    assert_eq!(incomplete["b"]["status"], "incomplete");
}

#[test]
fn trace_reporting_keeps_margins_and_whole_trace_consistency() {
    let reference = speech::TraceStep {
        frame_index: 0,
        token_top1: 3,
        token_top2: 2,
        duration_top1: 1,
        duration_top2: 0,
        token_logit1: 2.0,
        token_logit2: 1.0,
        duration_logit1: 2.0,
        duration_logit2: 1.0,
        emitted: true,
        cap_hit: false,
    };
    let mut actual = reference.clone();
    actual.token_top1 = 4;
    let report = trace_findings(&[reference], &[actual], &[[4, 0, 1]], 1, 0.01);
    assert_eq!(report["status"], "failed");
    assert_eq!(report["whole_trace_triples_consistent"], true);
    assert_eq!(report["first_difference"]["reference_token_margin"], 1.0);
    assert_eq!(report["first_difference"]["kind"], "argmax");
}

pub(super) fn open_case_journal() -> Result<File> {
    let path = env_path("ANE_PARAKEET_REPORT")?.with_extension("cases.jsonl");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(File::create(path)?)
}

pub(super) fn write_case_record(writer: &mut impl std::io::Write, row: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[test]
fn case_journal_retains_failed_and_later_successful_measurements() {
    let mut bytes = Vec::new();
    write_case_record(
        &mut bytes,
        &json!({"id": "first", "status": "grade-failed"}),
    )
    .unwrap();
    write_case_record(&mut bytes, &json!({"id": "second", "status": "passed"})).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    let rows: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["status"], "grade-failed");
    assert_eq!(rows[1]["id"], "second");
}

#[test]
fn timing_estimate_counts_parity_calls_and_marks_missing_rungs_unmeasured() {
    let runs = json!([{"encoder_seconds": 0.1, "decode_seconds": 0.2, "seconds": 0.3, "max_progress_gap_seconds": 0.01}]);
    let cases = vec![
        json!({"shape": 128, "stored_tensor": true, "mel_frames": 9, "decoder_work": 100,
        "timing": {"prepare_seconds": 0.02, "grading_seconds": 0.003,
        "runs": [runs[0],runs[0],runs[0]]}}),
        json!({"shape": 128, "stored_tensor": false, "mel_frames": 9, "decoder_work": 200,
        "timing": {"prepare_seconds": 0.03, "grading_seconds": 0.004, "runs": runs}}),
    ];
    let mut population = vec![
        json!({"rung": 128, "stored": true, "mel_frames": 9, "decoder_work": 100}),
        json!({"rung": 128, "stored": false, "mel_frames": 9, "decoder_work": 200}),
    ];
    let admissions =
        vec![json!({"shape": 128, "seconds": 0.5, "cache_hit_layers": 24, "compiled_layers": 0})];
    let report = timing_report(&cases, &population, &admissions);
    assert_eq!(report["complete"], true);
    assert_eq!(report["by_rung"]["128"]["population_engine_calls"], 4);
    assert!(
        (report["estimated_full_seconds_including_observed_setup"]
            .as_f64()
            .unwrap()
            - 1.757)
            .abs()
            < 1e-9
    );
    population.push(json!({"rung": 256, "stored": true, "mel_frames": 17, "decoder_work": 200}));
    let incomplete = timing_report(&cases, &population, &admissions);
    assert_eq!(incomplete["complete"], false);
    assert!(incomplete["estimated_full_seconds_including_observed_setup"].is_null());
}
