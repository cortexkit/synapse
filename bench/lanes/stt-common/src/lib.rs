pub mod artifact;
pub mod input;
pub mod measurement;
pub mod memory;
mod nvml;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use input::{InputEntry, Utterance};
use measurement::{percentile, validate_measurements, WorkerResult};
use serde::{Deserialize, Serialize};
use synapse_parity::speech::{grade_wer, load_transcribe_fixtures, SpeechFixture};

#[derive(Serialize, Deserialize)]
pub struct Request {
    pub utterances: Vec<Utterance>,
}

pub struct Corpus {
    pub request: Request,
    pub fixtures: BTreeMap<String, SpeechFixture>,
}

#[derive(Deserialize)]
struct CorpusDocument {
    cases: Vec<CorpusCase>,
}
#[derive(Deserialize)]
struct CorpusCase {
    id: String,
    split: String,
    ground_truth: String,
    sample_count: u64,
}

pub fn load_corpus(parity: &Path, wavs: &Path) -> Result<Corpus> {
    let manifest = synapse_parity::manifest::Manifest::load(&parity.join("models.json"))?;
    let fixtures: BTreeMap<_, _> = load_transcribe_fixtures(parity, &manifest)?
        .into_iter()
        .map(|fixture| (fixture.id.clone(), fixture))
        .collect();
    let corpus: CorpusDocument =
        serde_json::from_slice(&fs::read(parity.join("reference/parakeet-corpus.json"))?)?;
    ensure!(
        corpus.cases.len() == 1200,
        "expected the complete 1,200-utterance corpus"
    );
    let index: BTreeMap<String, InputEntry> =
        serde_json::from_slice(&fs::read(wavs.join("inputs.json"))?)?;
    let mut seen = BTreeSet::new();
    let mut utterances = Vec::new();
    for case in corpus.cases {
        ensure!(seen.insert(case.id.clone()), "duplicate corpus ID");
        let fixture = fixtures
            .get(&case.id)
            .context("corpus ID absent from sealed fixtures")?;
        ensure!(
            fixture.sample_count == case.sample_count
                && fixture.split.as_deref() == Some(&case.split)
                && fixture.ground_truth.as_deref() == Some(&case.ground_truth),
            "corpus and sealed fixture disagree: {}",
            case.id
        );
        let entry = index
            .get(&case.id)
            .with_context(|| format!("missing WAV: {}", case.id))?;
        utterances.push(Utterance {
            path: input::verify_audio(wavs, entry, case.sample_count)?,
            locale: locale(&case.split)?.into(),
            split: case.split,
            id: case.id,
            duration_seconds: case.sample_count as f64 / 16000.0,
        });
    }
    Ok(Corpus {
        request: Request { utterances },
        fixtures,
    })
}

fn locale(split: &str) -> Result<&'static str> {
    Ok(match split {
        "librispeech-test-clean" | "librispeech-test-other" => "en-US",
        "fleurs-de-test" => "de-DE",
        "fleurs-el-test" => "el-GR",
        "fleurs-es-test" => "es-ES",
        "fleurs-fr-test" => "fr-FR",
        "fleurs-it-test" => "it-IT",
        "fleurs-nl-test" => "nl-NL",
        "fleurs-pl-test" => "pl-PL",
        "fleurs-ru-test" => "ru-RU",
        _ => anyhow::bail!("unknown corpus split: {split}"),
    })
}

#[derive(Debug, Serialize)]
pub struct SplitScore {
    pub requested: usize,
    pub completed: usize,
    pub ground_truth_wer: Option<f64>,
    pub nvidia_fp32_wer: Option<f64>,
    pub audio_seconds: f64,
    pub inference_seconds: f64,
    pub real_time_factor: Option<f64>,
    pub latency_p50_seconds: Option<f64>,
    pub latency_p95_seconds: Option<f64>,
}

pub fn score(corpus: &Corpus, result: &WorkerResult) -> Result<BTreeMap<String, SplitScore>> {
    validate_measurements(result)?;
    let mut transcripts = BTreeMap::new();
    for transcript in &result.transcripts {
        ensure!(
            transcripts
                .insert(transcript.id.as_str(), transcript)
                .is_none(),
            "duplicate transcript ID: {}",
            transcript.id
        );
    }
    let requested: BTreeSet<_> = corpus
        .request
        .utterances
        .iter()
        .map(|u| u.id.as_str())
        .collect();
    ensure!(
        transcripts.keys().copied().collect::<BTreeSet<_>>() == requested,
        "missing or unexpected transcript IDs"
    );
    let mut truth_cases = Vec::new();
    let mut nvidia_cases = Vec::new();
    let mut splits: BTreeMap<String, SplitScore> = BTreeMap::new();
    let mut latencies: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for utterance in &corpus.request.utterances {
        let row = splits.entry(utterance.split.clone()).or_insert(SplitScore {
            requested: 0,
            completed: 0,
            ground_truth_wer: None,
            nvidia_fp32_wer: None,
            audio_seconds: 0.0,
            inference_seconds: 0.0,
            real_time_factor: None,
            latency_p50_seconds: None,
            latency_p95_seconds: None,
        });
        row.requested += 1;
        let transcript = transcripts[utterance.id.as_str()];
        if let (Some(text), Some(seconds)) = (&transcript.text, transcript.seconds) {
            let fixture = &corpus.fixtures[&utterance.id];
            let truth = fixture
                .ground_truth
                .as_deref()
                .context("missing ground truth")?;
            // grade_wer rejects candidates more than 0.005 worse than its
            // reference hypothesis, a certification gate for Synapse engines.
            // Supply the competitor text for both hypotheses against each truth
            // so benchmarking measures WER without imposing that accuracy gate.
            truth_cases.push((
                utterance.split.as_str(),
                truth,
                text.as_str(),
                text.as_str(),
            ));
            nvidia_cases.push((
                utterance.split.as_str(),
                fixture.text.as_str(),
                text.as_str(),
                text.as_str(),
            ));
            row.completed += 1;
            row.audio_seconds += utterance.duration_seconds;
            row.inference_seconds += seconds;
            latencies
                .entry(utterance.split.clone())
                .or_default()
                .push(seconds);
        }
    }
    let truth = grade_wer(truth_cases)?;
    let nvidia = grade_wer(nvidia_cases)?;
    for (split, row) in &mut splits {
        row.ground_truth_wer = truth.get(split).map(|rates| rates[1]);
        row.nvidia_fp32_wer = nvidia.get(split).map(|rates| rates[1]);
        row.real_time_factor =
            (row.audio_seconds > 0.0).then(|| row.inference_seconds / row.audio_seconds);
        let times = latencies.get(split).map(Vec::as_slice).unwrap_or_default();
        row.latency_p50_seconds = percentile(times, 0.5);
        row.latency_p95_seconds = percentile(times, 0.95);
    }
    Ok(splits)
}
