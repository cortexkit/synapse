//! Parakeet's reference contract and transcription grading. These rules are
//! deliberately separate from the legacy embedding/reranking admission block.

use std::collections::BTreeMap;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use unicode_casefold::UnicodeCaseFold;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;

use crate::manifest::Architecture;
use crate::{perr, Result};

pub const MODEL: &str = "parakeet-tdt-0.6b-v3";
pub const NORMALIZER: &str = "synapse-wer-norm-v1";
pub const BLANK: i32 = 8192;
pub const DURATIONS: [i32; 5] = [0, 1, 2, 3, 4];
pub const CAP: usize = 10;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechProfile {
    pub max_duration_ms: u64,
    pub encoder_rungs: Vec<u32>,
    pub tau_feat: Option<f64>,
    pub tau_resample: Option<f64>,
    pub tau_margin: Option<f64>,
    pub confirmations: Value,
    pub self_check: Artifact,
}

impl SpeechProfile {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("tau_feat", self.tau_feat),
            ("tau_resample", self.tau_resample),
            ("tau_margin", self.tau_margin),
        ] {
            if !value.is_some_and(|x| x.is_finite() && x > 0.0) {
                return Err(perr!("Parakeet profile requires positive finite {name}"));
            }
        }
        if self.max_duration_ms == 0
            || self.max_duration_ms > 655360
            || self.encoder_rungs != [128, 256, 512, 1024, 2048, 4096, 8192]
        {
            return Err(perr!("invalid Parakeet duration envelope or encoder rungs"));
        }
        if self
            .confirmations
            .get("transformers_version")
            .and_then(Value::as_str)
            != Some("5.16.1")
        {
            return Err(perr!(
                "Parakeet confirmations must name transformers 5.16.1"
            ));
        }
        Ok(())
    }
}

/// A committed artifact, or a content-addressed artifact in the caller's mirror.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_path: Option<String>,
    pub sha256: String,
    pub bytes: u64,
}

impl Artifact {
    pub fn read(&self, parity_dir: &Path, mirror: &Path) -> Result<Vec<u8>> {
        let (root, relative) = match (&self.path, &self.cache_path) {
            (Some(path), None) => (parity_dir, path),
            (None, Some(path)) => (mirror, path),
            _ => return Err(perr!("artifact must name exactly one path or cache_path")),
        };
        if Path::new(relative)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(perr!(
                "artifact path must be relative and contain no traversal"
            ));
        }
        let bytes = std::fs::read(root.join(relative))
            .map_err(|e| perr!("read artifact {relative}: {e}"))?;
        let digest = crate::canonical::sha256_hex(&bytes);
        if bytes.len() as u64 != self.bytes || digest != self.sha256 {
            return Err(perr!(
                "artifact {relative} differs from its seal: got {} bytes sha256:{digest}, expected {} bytes sha256:{}",
                bytes.len(), self.bytes, self.sha256
            ));
        }
        Ok(bytes)
    }
}

/// Actual mel frames, valid mel frames, valid encoder frames at 16 kHz.
pub fn frame_counts(samples: u64) -> (u64, u64, u64) {
    let mel = samples / 160;
    (mel + 1, mel, mel.div_ceil(8))
}

/// Exact name/shape inventory, independent of the downloaded tensor index.
pub fn expected_tensors(arch: &Architecture) -> Result<BTreeMap<String, Vec<u64>>> {
    let h = arch.int("hidden_size")?;
    let d = arch.int("decoder_hidden_size")?;
    let v = arch.int("vocab_size")?;
    let f = arch.int("intermediate_size")?;
    let channels = arch.int("subsampling_conv_channels")?;
    let heads = arch.int("num_attention_heads")?;
    if arch.int("subsampling_factor")? != 8
        || arch.int("subsampling_conv_stride")? != 2
        || arch.int("subsampling_conv_kernel_size")? != 3
        || h % heads != 0
        || arch.boolean("attention_bias")?
        || arch.boolean("convolution_bias")?
    {
        return Err(perr!(
            "unsupported Parakeet subsampling or bias configuration"
        ));
    }
    let mut tensors = BTreeMap::new();
    let mut put = |name: String, shape: Vec<u64>| {
        tensors.insert(name, shape);
    };
    put("decoder.embedding.weight".into(), vec![v, d]);
    for layer in 0..arch.int("num_decoder_layers")? {
        for kind in ["ih", "hh"] {
            put(
                format!("decoder.lstm.weight_{kind}_l{layer}"),
                vec![4 * d, d],
            );
            put(format!("decoder.lstm.bias_{kind}_l{layer}"), vec![4 * d]);
        }
    }
    for (name, output, input) in [
        ("decoder.decoder_projector", d, d),
        ("encoder_projector", d, h),
        ("joint.head", v + 5, d),
    ] {
        put(format!("{name}.weight"), vec![output, input]);
        put(format!("{name}.bias"), vec![output]);
    }
    for (index, input, kernel) in [
        (0, 1, 3),
        (2, 1, 3),
        (3, channels, 1),
        (5, 1, 3),
        (6, channels, 1),
    ] {
        put(
            format!("encoder.subsampling.layers.{index}.weight"),
            vec![channels, input, kernel, kernel],
        );
        put(
            format!("encoder.subsampling.layers.{index}.bias"),
            vec![channels],
        );
    }
    put(
        "encoder.subsampling.linear.weight".into(),
        vec![h, channels * arch.int("num_mel_bins")?.div_ceil(8)],
    );
    put("encoder.subsampling.linear.bias".into(), vec![h]);
    for layer in 0..arch.int("num_hidden_layers")? {
        let l = format!("encoder.layers.{layer}");
        for norm in [
            "norm_conv",
            "norm_feed_forward1",
            "norm_feed_forward2",
            "norm_out",
            "norm_self_att",
        ] {
            for suffix in ["weight", "bias"] {
                put(format!("{l}.{norm}.{suffix}"), vec![h]);
            }
        }
        for ff in [1, 2] {
            put(format!("{l}.feed_forward{ff}.linear1.weight"), vec![f, h]);
            put(format!("{l}.feed_forward{ff}.linear2.weight"), vec![h, f]);
        }
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"] {
            put(format!("{l}.self_attn.{projection}.weight"), vec![h, h]);
        }
        for bias in ["bias_u", "bias_v"] {
            put(format!("{l}.self_attn.{bias}"), vec![heads, h / heads]);
        }
        put(
            format!("{l}.conv.pointwise_conv1.weight"),
            vec![2 * h, h, 1],
        );
        put(format!("{l}.conv.pointwise_conv2.weight"), vec![h, h, 1]);
        put(
            format!("{l}.conv.depthwise_conv.weight"),
            vec![h, 1, arch.int("conv_kernel_size")?],
        );
        for suffix in ["weight", "bias", "running_mean", "running_var"] {
            put(format!("{l}.conv.norm.{suffix}"), vec![h]);
        }
        put(format!("{l}.conv.norm.num_batches_tracked"), vec![]);
    }
    Ok(tensors)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceStep {
    pub frame_index: i32,
    pub token_top1: i32,
    pub token_top2: i32,
    pub duration_top1: i32,
    pub duration_top2: i32,
    pub token_logit1: f32,
    pub token_logit2: f32,
    pub duration_logit1: f32,
    pub duration_logit2: f32,
    pub emitted: bool,
    pub cap_hit: bool,
}

pub fn decode_trace(bytes: &[u8]) -> Result<Vec<TraceStep>> {
    if bytes.len() % 40 != 0 {
        return Err(perr!("trace length is not steps * 40"));
    }
    bytes
        .chunks_exact(40)
        .map(|record| {
            let i = |offset| i32::from_le_bytes(record[offset..offset + 4].try_into().unwrap());
            let f = |offset| f32::from_le_bytes(record[offset..offset + 4].try_into().unwrap());
            let flags = u32::from_le_bytes(record[36..40].try_into().unwrap());
            if flags & !3 != 0 {
                return Err(perr!("trace has unknown flags"));
            }
            Ok(TraceStep {
                frame_index: i(0),
                token_top1: i(4),
                token_top2: i(8),
                duration_top1: i(12),
                duration_top2: i(16),
                token_logit1: f(20),
                token_logit2: f(24),
                duration_logit1: f(28),
                duration_logit2: f(32),
                emitted: flags & 1 != 0,
                cap_hit: flags & 2 != 0,
            })
        })
        .collect()
}

pub type Triple = [i32; 3];

pub fn implied_triples(trace: &[TraceStep], valid_frames: u64) -> Result<Vec<Triple>> {
    let mut expected_frame = 0i64;
    let mut evaluations = 0;
    let mut triples = Vec::new();
    for step in trace {
        if expected_frame >= valid_frames as i64
            || i64::from(step.frame_index) != expected_frame
            || !(0..=BLANK).contains(&step.token_top1)
            || !(0..=BLANK).contains(&step.token_top2)
            || step.token_top1 == step.token_top2
            || !(0..5).contains(&step.duration_top1)
            || !(0..5).contains(&step.duration_top2)
            || step.duration_top1 == step.duration_top2
            || step.emitted != (step.token_top1 != BLANK)
            || ![
                step.token_logit1,
                step.token_logit2,
                step.duration_logit1,
                step.duration_logit2,
            ]
            .iter()
            .all(|v| v.is_finite())
            || step.token_logit1 < step.token_logit2
            || step.duration_logit1 < step.duration_logit2
            || (step.token_logit1 == step.token_logit2 && step.token_top1 > step.token_top2)
            || (step.duration_logit1 == step.duration_logit2
                && step.duration_top1 > step.duration_top2)
        {
            return Err(perr!("invalid or nonfinite decode trace step"));
        }
        evaluations += 1;
        let d = DURATIONS[step.duration_top1 as usize];
        if step.emitted {
            triples.push([step.token_top1, step.frame_index, d]);
        }
        let advance = if step.emitted { d } else { d.max(1) };
        let cap_hit = evaluations == CAP && advance == 0;
        if step.cap_hit != cap_hit {
            return Err(perr!("trace cap flag disagrees with bounded rule"));
        }
        if advance > 0 || cap_hit {
            expected_frame += i64::from(advance.max(1));
            evaluations = 0;
        }
    }
    if expected_frame < valid_frames as i64 {
        return Err(perr!("truncated trace: encoder not exhausted"));
    }
    Ok(triples)
}

/// Returns whether the first differing decision is margin-exempt. Later steps
/// cannot rescue a frame-index mismatch, a length mismatch, or invented triples.
pub fn grade_trace(
    reference: &[TraceStep],
    actual: &[TraceStep],
    triples: &[Triple],
    valid_frames: u64,
    tau_margin: f64,
) -> Result<bool> {
    if implied_triples(actual, valid_frames)? != triples {
        return Err(perr!(
            "emitted triples are inconsistent with the whole engine trace"
        ));
    }
    implied_triples(reference, valid_frames)?;
    for index in 0..reference.len().max(actual.len()) {
        let (Some(r), Some(a)) = (reference.get(index), actual.get(index)) else {
            return Err(perr!("trace length differs at step {index}"));
        };
        if r.frame_index != a.frame_index {
            return Err(perr!("trace frame index differs at step {index}"));
        }
        let token = r.token_top1 != a.token_top1;
        let duration = r.duration_top1 != a.duration_top1;
        if token || duration {
            if (!token || f64::from(r.token_logit1 - r.token_logit2) < tau_margin)
                && (!duration || f64::from(r.duration_logit1 - r.duration_logit2) < tau_margin)
            {
                return Ok(true);
            }
            return Err(perr!("trace argmax differs outside margin at step {index}"));
        }
    }
    Ok(false)
}

pub fn grade_encoder(reference: &[f32], actual: &[f32], frames: usize) -> Result<f64> {
    if frames == 0
        || reference.len() != frames * 1024
        || actual.len() != reference.len()
        || !reference.iter().chain(actual).all(|v| v.is_finite())
    {
        return Err(perr!("missing, malformed or nonfinite encoder observation"));
    }
    let mut minimum = 1.0f64;
    for (r, a) in reference.chunks_exact(1024).zip(actual.chunks_exact(1024)) {
        let mut dot = 0.0;
        let mut rr = 0.0;
        let mut aa = 0.0;
        for (&x, &y) in r.iter().zip(a) {
            let x = f64::from(x);
            let y = f64::from(y);
            dot += x * y;
            rr += x * x;
            aa += y * y;
        }
        let cosine = dot / (rr * aa).sqrt();
        if !cosine.is_finite() || cosine < 0.999 {
            return Err(perr!("encoder per-frame cosine gate failed: {cosine}"));
        }
        minimum = minimum.min(cosine);
    }
    Ok(minimum)
}

pub fn normalize_words(text: &str) -> Vec<String> {
    text.nfkc()
        .case_fold()
        .filter(|c| {
            !matches!(
                get_general_category(*c),
                GeneralCategory::ConnectorPunctuation
                    | GeneralCategory::DashPunctuation
                    | GeneralCategory::ClosePunctuation
                    | GeneralCategory::FinalPunctuation
                    | GeneralCategory::InitialPunctuation
                    | GeneralCategory::OpenPunctuation
                    | GeneralCategory::OtherPunctuation
            )
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

pub fn word_edits(truth: &[String], actual: &[String]) -> usize {
    let mut previous: Vec<usize> = (0..=actual.len()).collect();
    for (i, expected) in truth.iter().enumerate() {
        let mut row = vec![i + 1];
        for (j, word) in actual.iter().enumerate() {
            row.push(
                (previous[j] + usize::from(expected != word))
                    .min(previous[j + 1] + 1)
                    .min(row[j] + 1),
            );
        }
        previous = row;
    }
    previous[actual.len()]
}

/// Accumulate raw edit counts per split, not a mean of per-utterance WERs.
pub fn grade_wer<'a>(
    cases: impl IntoIterator<Item = (&'a str, &'a str, &'a str, &'a str)>,
) -> Result<BTreeMap<String, [f64; 2]>> {
    let mut counts: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    for (split, truth, reference, actual) in cases {
        let truth = normalize_words(truth);
        let row = counts.entry(split.to_string()).or_default();
        row[0] += truth.len();
        row[1] += word_edits(&truth, &normalize_words(reference));
        row[2] += word_edits(&truth, &normalize_words(actual));
    }
    counts
        .into_iter()
        .map(|(split, [words, reference, actual])| {
            if words == 0 {
                return Err(perr!("split {split} has no ground-truth words"));
            }
            let rates = [
                reference as f64 / words as f64,
                actual as f64 / words as f64,
            ];
            if rates[1] - rates[0] > 0.005 {
                return Err(perr!(
                    "split {split} exceeds WER delta of 0.5 absolute points"
                ));
            }
            Ok((split, rates))
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechRefusal {
    pub code: String,
    /// duration_ms, max_duration_ms, valid_encoder_frames, max_encoder_frames.
    pub details: [u64; 4],
}

/// Above-limit cases require an exact refusal before any engine work. The caller
/// supplies the pinned media contract's duration-limit code (or invalid_params
/// when it defines none); a fixture's duration is never truncated.
pub fn grade_applicability(
    profile: &SpeechProfile,
    expected_refusal_code: &str,
    duration_ms: u64,
    samples_16k: u64,
    observed_frames: Option<u64>,
    refusal: Option<&SpeechRefusal>,
    engine_started: bool,
) -> Result<bool> {
    profile.validate()?;
    if expected_refusal_code.is_empty() {
        return Err(perr!("expected duration refusal code is missing"));
    }
    let frames = frame_counts(samples_16k).2;
    let max_frames = frame_counts(profile.max_duration_ms * 16).2;
    let above = duration_ms > profile.max_duration_ms || frames > max_frames;
    if above {
        if engine_started
            || observed_frames.is_some()
            || !refusal.is_some_and(|refusal| {
                refusal.code == expected_refusal_code
                    && refusal.details == [duration_ms, profile.max_duration_ms, frames, max_frames]
            })
        {
            return Err(perr!("above-limit fixture was not refused before engine work with the contract code and exact duration details"));
        }
        Ok(false)
    } else if refusal.is_some() || observed_frames != Some(frames) {
        Err(perr!(
            "missing observation, in-limit refusal, or valid-frame-count mismatch"
        ))
    } else {
        Ok(true)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpeechFixture {
    pub id: String,
    pub sample_count: u64,
    pub duration_ms: u64,
    pub valid_encoder_frames: u64,
    pub trace: Artifact,
    pub encoder: Option<Artifact>,
    pub text: String,
    pub split: Option<String>,
    pub ground_truth: Option<String>,
}

pub fn load_transcribe_fixtures(
    root: &Path,
    manifest: &crate::manifest::Manifest,
) -> Result<Vec<SpeechFixture>> {
    let index: Value = serde_json::from_slice(
        &std::fs::read(root.join("fixtures/index.json"))
            .map_err(|e| perr!("read fixture index: {e}"))?,
    )
    .map_err(|e| perr!("fixture index: {e}"))?;
    let id = format!("{MODEL}.ref-v1.transformers-5.16.1.seed-0");
    let entry = &index[&id];
    if entry["model"] != MODEL || entry["transformers_version"] != "5.16.1" || entry["seed"] != 0 {
        return Err(perr!("speech fixture index metadata mismatch"));
    }
    let seal: Artifact = serde_json::from_value(serde_json::json!({"path": entry["path"],
        "sha256": entry["sha256"], "bytes": entry["bytes"]}))
    .map_err(|e| perr!("fixture seal: {e}"))?;
    let document: Value = serde_json::from_slice(&seal.read(root, root)?)
        .map_err(|e| perr!("speech fixture document: {e}"))?;
    let model = manifest.model(MODEL)?;
    if document["fixture_set_id"] != id
        || document["model"] != MODEL
        || document["hf_revision"] != model.hf_revision
        || document["checkpoint_digest"] != model.checkpoint_digest
        || document["tokenizer_digest"] != model.tokenizer_digest
    {
        return Err(perr!(
            "speech fixture checkpoint/tokenizer binding mismatch"
        ));
    }
    let cases = document["cases"]
        .as_array()
        .ok_or_else(|| perr!("speech fixture cases missing"))?;
    for case in cases {
        let case_id = case["id"]
            .as_str()
            .ok_or_else(|| perr!("fixture id missing"))?;
        for key in ["trace", "encoder", "log_mel"] {
            if entry["artifacts"][case_id][key] != case[key] {
                return Err(perr!("speech tensor index seal differs: {case_id} {key}"));
            }
        }
    }
    serde_json::from_value(document["cases"].clone())
        .map_err(|e| perr!("speech fixture schema: {e}"))
}

#[derive(Clone, Debug, Default)]
pub struct SpeechObservation {
    pub valid_encoder_frames: Option<u64>,
    pub refusal: Option<SpeechRefusal>,
    pub engine_started: bool,
    pub encoder_output: Option<Vec<f32>>,
    pub trace: Vec<TraceStep>,
    pub triples: Vec<Triple>,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct SpeechReport {
    pub graded: Vec<String>,
    pub expected_refusals: Vec<String>,
    pub margin_exempt: Vec<String>,
    pub split_wer: BTreeMap<String, [f64; 2]>,
}

/// Grade all in-envelope fixtures and record out-of-envelope refusals separately.
/// Even margin-exempt traces must pass the encoder gate and whole-trace triples.
pub fn evaluate_transcribe(
    profile: &SpeechProfile,
    fixtures: &[SpeechFixture],
    observations: &BTreeMap<String, SpeechObservation>,
    parity_dir: &Path,
    mirror: &Path,
    expected_refusal_code: &str,
) -> Result<SpeechReport> {
    profile.validate()?;
    if fixtures.is_empty() {
        return Err(perr!("transcription fixture set is empty"));
    }
    let mut report = SpeechReport {
        graded: vec![],
        expected_refusals: vec![],
        margin_exempt: vec![],
        split_wer: BTreeMap::new(),
    };
    let mut wer = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for fixture in fixtures {
        if !ids.insert(&fixture.id)
            || fixture.valid_encoder_frames != frame_counts(fixture.sample_count).2
        {
            return Err(perr!(
                "duplicate fixture or reference frame-count mismatch: {}",
                fixture.id
            ));
        }
        if !fixture.id.starts_with("ladder-") && fixture.duration_ms > profile.max_duration_ms {
            return Err(perr!(
                "lane maximum does not cover every non-ladder fixture"
            ));
        }
        let actual = observations
            .get(&fixture.id)
            .ok_or_else(|| perr!("missing observation: {}", fixture.id))?;
        if !grade_applicability(
            profile,
            expected_refusal_code,
            fixture.duration_ms,
            fixture.sample_count,
            actual.valid_encoder_frames,
            actual.refusal.as_ref(),
            actual.engine_started,
        )? {
            if actual.encoder_output.is_some()
                || !actual.trace.is_empty()
                || !actual.triples.is_empty()
                || !actual.text.is_empty()
            {
                return Err(perr!("above-limit refusal carries inference output"));
            }
            report.expected_refusals.push(fixture.id.clone());
            continue;
        }
        let actual_encoder = actual
            .encoder_output
            .as_deref()
            .ok_or_else(|| perr!("missing encoder observation: {}", fixture.id))?;
        if actual_encoder.len() != fixture.valid_encoder_frames as usize * 1024
            || actual_encoder.iter().any(|x| !x.is_finite())
        {
            return Err(perr!(
                "malformed or nonfinite encoder observation: {}",
                fixture.id
            ));
        }
        if let Some(encoder) = &fixture.encoder {
            let raw = encoder.read(parity_dir, mirror)?;
            if raw.len() % 4 != 0 {
                return Err(perr!("encoder artifact is not f32le"));
            }
            let reference: Vec<f32> = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            grade_encoder(
                &reference,
                actual_encoder,
                fixture.valid_encoder_frames as usize,
            )?;
        }
        let reference = decode_trace(&fixture.trace.read(parity_dir, mirror)?)?;
        if grade_trace(
            &reference,
            &actual.trace,
            &actual.triples,
            fixture.valid_encoder_frames,
            profile.tau_margin.unwrap(),
        )? {
            report.margin_exempt.push(fixture.id.clone());
        }
        if let (Some(split), Some(truth)) = (&fixture.split, &fixture.ground_truth) {
            wer.push((
                split.as_str(),
                truth.as_str(),
                fixture.text.as_str(),
                actual.text.as_str(),
            ));
        } else if fixture.split.is_some() || fixture.ground_truth.is_some() {
            return Err(perr!("WER fixture must name both split and ground truth"));
        }
        report.graded.push(fixture.id.clone());
    }
    if observations.keys().any(|id| !ids.contains(id)) {
        return Err(perr!("observation has no reference fixture"));
    }
    report.split_wer = grade_wer(wer)?;
    Ok(report)
}
