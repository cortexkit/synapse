use super::*;
use crate::measurement::Transcript;

fn fixture(id: &str, truth: &str, reference: &str) -> SpeechFixture {
    serde_json::from_value(serde_json::json!({
        "id": id, "sample_count": 16000, "duration_ms": 1000,
        "valid_encoder_frames": 13,
        "trace": {"path": "unused", "sha256": "0".repeat(64), "bytes": 0},
        "encoder": null, "text": reference,
        "split": "test", "ground_truth": truth,
    }))
    .unwrap()
}

fn result(texts: &[(&str, &str)]) -> WorkerResult {
    WorkerResult {
        engine: "test".into(),
        version: "1".into(),
        commit: "abc".into(),
        backend: "test".into(),
        cold_start_seconds: 0.5,
        warm_wall_seconds: 1.0,
        peak_rss_bytes: 1,
        gpu_peak_bytes: None,
        energy_joules: None,
        transcripts: texts
            .iter()
            .map(|(id, text)| Transcript {
                id: (*id).into(),
                text: Some((*text).into()),
                seconds: Some(0.1),
                error: None,
            })
            .collect(),
    }
}

fn corpus(fixtures: Vec<SpeechFixture>) -> Corpus {
    Corpus {
        request: Request {
            utterances: fixtures
                .iter()
                .map(|f| Utterance {
                    id: f.id.clone(),
                    path: "unused".into(),
                    split: "test".into(),
                    locale: "en-US".into(),
                    duration_seconds: 1.0,
                })
                .collect(),
        },
        fixtures: fixtures.into_iter().map(|f| (f.id.clone(), f)).collect(),
    }
}

#[test]
fn stt_scoring_uses_parity_unicode_normalization_and_micro_aggregation() {
    // Literal expectations protect the normalization contract, including deleted
    // (not space-replaced) punctuation, full case folding and compatibility forms.
    let examples = [
        ("Straße — ＡＢＣ!", vec!["strasse", "abc"]),
        ("Don't re-enter foo_bar.", vec!["dont", "reenter", "foobar"]),
        ("ÉCOLE\u{a0}ΣΊΓΜΑ", vec!["école", "σίγμα"]),
    ];
    for (input, expected) in examples {
        assert_eq!(synapse_parity::speech::normalize_words(input), expected);
        let joined = expected.join(" ");
        let c = corpus(vec![fixture("a", input, &joined)]);
        let scores = score(&c, &result(&[("a", &joined)])).unwrap();
        assert_eq!(scores["test"].ground_truth_wer, Some(0.0));
        assert_eq!(scores["test"].nvidia_fp32_wer, Some(0.0));
    }
    let c = corpus(vec![
        fixture("a", "one", "one"),
        fixture("b", "two three four", "two three"),
    ]);
    let scores = score(&c, &result(&[("a", "wrong"), ("b", "two three four")])).unwrap();
    assert_eq!(scores["test"].ground_truth_wer, Some(0.25));
    assert_eq!(scores["test"].nvidia_fp32_wer, Some(2.0 / 3.0));
}

#[test]
fn stt_scoring_rejects_duplicate_missing_and_unexpected_ids() {
    let c = corpus(vec![fixture("a", "one", "one")]);
    assert!(score(&c, &result(&[])).is_err());
    assert!(score(&c, &result(&[("a", "one"), ("a", "one")])).is_err());
    assert!(score(&c, &result(&[("other", "one")])).is_err());
}

#[test]
fn stt_scoring_reports_unsupported_coverage_without_fake_empty_predictions() {
    let c = corpus(vec![fixture("a", "one", "one"), fixture("b", "two", "two")]);
    let mut r = result(&[("a", "one")]);
    r.transcripts.push(Transcript {
        id: "b".into(),
        text: None,
        seconds: None,
        error: Some("unsupported locale".into()),
    });
    let scores = score(&c, &r).unwrap();
    assert_eq!(scores["test"].requested, 2);
    assert_eq!(scores["test"].completed, 1);
    assert_eq!(scores["test"].ground_truth_wer, Some(0.0));
}

#[test]
fn stt_corpus_manifest_has_expected_splits_and_counts() {
    let document: CorpusDocument = serde_json::from_str(include_str!(
        "../../../parity/reference/parakeet-corpus.json"
    ))
    .unwrap();
    let mut counts = BTreeMap::new();
    for case in &document.cases {
        locale(&case.split).unwrap();
        *counts.entry(&case.split).or_insert(0) += 1;
    }
    assert_eq!(document.cases.len(), 1200);
    assert_eq!(counts.len(), 10);
    for (split, count) in counts {
        assert_eq!(
            count,
            if split.starts_with("librispeech") {
                200
            } else {
                100
            }
        );
    }
}
