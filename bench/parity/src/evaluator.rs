//! Backend-independent comparisons against sealed CPU reference fixtures.
//!
//! Callers acquire outputs through the module API, preserving fixture IDs and
//! API-domain scores. No reference generation or backend inference happens here.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::canonical::sha256_hex;
use crate::manifest::{Family, Manifest, Operation, ToleranceClass};
use crate::{perr, Result};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Output {
    Score(f64),
    Embedding(Vec<f64>),
}

#[derive(Clone, Debug, Deserialize)]
pub struct Case {
    pub id: String,
    pub category: String,
    pub input_ids: Vec<u32>,
    pub output: Output,
    pub pool: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Reference {
    transformers_version: String,
    seed: u64,
    device: String,
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct FixtureDocument {
    fixture_set_id: String,
    model: String,
    operation: Operation,
    hf_revision: String,
    checkpoint_digest: String,
    tokenizer_digest: String,
    reference: Reference,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct IndexEntry {
    path: String,
    sha256: String,
    model: String,
    transformers_version: String,
    seed: u64,
}

/// References whose bytes match the supplied fixture digest and whose model,
/// checkpoint, tokenizer, reference version and seed match the manifest.
#[derive(Debug)]
pub struct FixtureSet {
    document: FixtureDocument,
}

pub fn fixture_set_id(manifest: &Manifest, model: &str) -> String {
    format!(
        "{model}.ref-v1.transformers-{}.seed-{}",
        manifest.reference.reference_transformers_version, manifest.reference.reference_seed
    )
}

impl FixtureSet {
    pub fn load(root: &Path, manifest: &Manifest, model: &str) -> Result<Self> {
        let index = std::fs::read(root.join("fixtures/index.json"))
            .map_err(|e| perr!("fixture index: {e}"))?;
        let index: BTreeMap<String, IndexEntry> =
            serde_json::from_slice(&index).map_err(|e| perr!("fixture index schema: {e}"))?;
        let id = fixture_set_id(manifest, model);
        let entry = index
            .get(&id)
            .ok_or_else(|| perr!("fixture_set_mismatch: missing {id}"))?;
        if entry.model != model
            || entry.transformers_version != manifest.reference.reference_transformers_version
            || entry.seed != manifest.reference.reference_seed
            || Path::new(&entry.path)
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(perr!("fixture_set_mismatch: index entry {id}"));
        }
        let bytes = std::fs::read(root.join(&entry.path))
            .map_err(|e| perr!("fixture {}: {e}", entry.path))?;
        Self::from_bytes(manifest, model, &bytes, &entry.sha256)
    }

    /// `expected_digest` must come from the committed fixture index, never from
    /// the backend output or from hashing the bytes being verified.
    pub fn from_bytes(
        manifest: &Manifest,
        model: &str,
        bytes: &[u8],
        expected_digest: &str,
    ) -> Result<Self> {
        if sha256_hex(bytes) != expected_digest {
            return Err(perr!("fixture_digest_mismatch: {model}"));
        }
        let document: FixtureDocument =
            serde_json::from_slice(bytes).map_err(|e| perr!("fixture schema: {e}"))?;
        let pinned = manifest.model(model)?;
        if document.fixture_set_id != fixture_set_id(manifest, model)
            || document.model != model
            || document.operation != pinned.operation
            || document.hf_revision != pinned.hf_revision
            || document.checkpoint_digest != pinned.checkpoint_digest
            || document.tokenizer_digest != pinned.tokenizer_digest
            || document.reference.transformers_version
                != manifest.reference.reference_transformers_version
            || document.reference.seed != manifest.reference.reference_seed
            || document.reference.device != "cpu"
            || document.reference.dtype != "fp32"
        {
            return Err(perr!("fixture_set_mismatch: {model}"));
        }
        let mut ids = BTreeSet::new();
        if document.cases.is_empty()
            || document.cases.iter().any(|c| {
                !ids.insert(&c.id)
                    || c.input_ids.is_empty()
                    || c.input_ids.len() > 8192
                    || match (&c.output, pinned.operation) {
                        (Output::Score(s), Operation::Rerank) => !s.is_finite(),
                        (Output::Embedding(v), Operation::Embed) => {
                            v.len() != pinned.output.dimension as usize
                                || v.iter().any(|x| !x.is_finite())
                                || norm(v) == 0.0
                        }
                        _ => true,
                    }
            })
        {
            return Err(perr!("invalid_fixture_cases: {model}"));
        }
        Ok(Self { document })
    }

    pub fn cases(&self) -> &[Case] {
        &self.document.cases
    }
}

/// Runtime provenance is required for Qwen reranking: scores alone cannot
/// distinguish a wrong template or swapped readout from numeric drift.
#[derive(Clone, Debug)]
pub struct ObservedCase {
    pub output: Output,
    pub input_ids: Vec<u32>,
    /// `(yes, no)` token IDs actually used by the backend readout.
    pub readout: Option<(u32, u32)>,
}

#[derive(Debug, Default, Serialize)]
pub struct Gates {
    pub complete: bool,
    pub dimension: bool,
    pub finite: bool,
    pub unit_norm: bool,
    pub cosine: bool,
    pub score: bool,
    pub ranking: bool,
    pub coverage_8192: bool,
    pub coverage: bool,
    pub semantics: bool,
}

#[derive(Debug, Default, Serialize)]
pub struct Metrics {
    pub min_cosine: Option<f64>,
    pub max_norm_error: Option<f64>,
    pub max_absolute_error: Option<f64>,
    pub pools: BTreeMap<String, Ranking>,
}

#[derive(Debug, Serialize)]
pub struct Evaluation {
    pub model: String,
    pub operation: Operation,
    pub profile_id: String,
    pub fingerprint: String,
    pub fixture_set_id: String,
    pub tolerance_class: ToleranceClass,
    pub completed_fixtures: usize,
    pub expected_fixtures: usize,
    pub metrics: Metrics,
    pub gates: Gates,
    pub failures: Vec<String>,
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[derive(Debug, Serialize)]
pub struct Ranking {
    pub concordant: usize,
    pub discordant: usize,
    pub backend_ties: usize,
    pub tau: f64,
    pub top10: bool,
    pub passed: bool,
}

/// Filter pairs by their reference gap directly, never by transitive tie groups.
pub fn ranking(reference: &[f64], actual: &[f64]) -> Result<Ranking> {
    if reference.len() != actual.len()
        || reference.is_empty()
        || reference.iter().chain(actual).any(|x| !x.is_finite())
    {
        return Err(perr!("ranking requires equal nonempty finite score arrays"));
    }
    let mut order: Vec<usize> = (0..reference.len()).collect();
    order.sort_by(|&a, &b| reference[b].total_cmp(&reference[a]).then(a.cmp(&b)));
    let top: BTreeSet<usize> = order.into_iter().take(10).collect();
    let (mut c, mut d, mut ties, mut top10) = (0, 0, 0, true);
    for a in 0..reference.len() {
        for b in a + 1..reference.len() {
            if (reference[a] - reference[b]).abs() < 0.01 {
                continue;
            }
            let (hi, lo) = if reference[a] > reference[b] {
                (a, b)
            } else {
                (b, a)
            };
            if actual[hi] > actual[lo] {
                c += 1;
            } else if actual[hi] < actual[lo] {
                d += 1;
            } else {
                ties += 1;
            }
            if top.contains(&hi) && actual[hi] <= actual[lo] {
                top10 = false;
            }
        }
    }
    let total = c + d + ties;
    let tau = if total == 0 {
        1.0
    } else {
        (c as f64 - d as f64) / total as f64
    };
    Ok(Ranking {
        concordant: c,
        discordant: d,
        backend_ties: ties,
        tau,
        top10,
        passed: top10 && tau >= 0.99,
    })
}

pub fn evaluate(
    manifest: &Manifest,
    profile_id: &str,
    fingerprint: &str,
    fixtures: &FixtureSet,
    outputs: &BTreeMap<String, ObservedCase>,
) -> Result<Evaluation> {
    let profile = manifest
        .profiles
        .get(profile_id)
        .ok_or_else(|| perr!("unknown profile {profile_id}"))?;
    let model = manifest.model(&profile.model)?;
    let doc = &fixtures.document;
    if doc.model != profile.model || doc.fixture_set_id != fixture_set_id(manifest, &profile.model)
    {
        return Err(perr!("fixture_set_mismatch: {profile_id}"));
    }
    let mut result = Evaluation {
        model: profile.model.clone(),
        operation: model.operation,
        profile_id: profile_id.into(),
        fingerprint: fingerprint.into(),
        fixture_set_id: doc.fixture_set_id.clone(),
        tolerance_class: profile.rerank_tolerance_class,
        completed_fixtures: 0,
        expected_fixtures: doc.cases.len(),
        metrics: Metrics::default(),
        gates: Gates::default(),
        failures: vec![],
    };
    if outputs.is_empty() {
        result.failures.push("no_output".into());
        return Ok(result);
    }
    let mut gates = Gates {
        complete: outputs.len() == doc.cases.len(),
        dimension: true,
        finite: true,
        unit_norm: true,
        cosine: true,
        score: true,
        ranking: true,
        semantics: true,
        ..Gates::default()
    };
    let mut pools: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut categories = BTreeSet::new();
    let mut min_cosine: f64 = 1.0;
    let mut max_norm: f64 = 0.0;
    let mut max_error: f64 = 0.0;
    for case in &doc.cases {
        let Some(observed) = outputs.get(&case.id) else {
            gates.complete = false;
            continue;
        };
        result.completed_fixtures += 1;
        categories.insert(case.category.as_str());
        if case.input_ids.len() == 8192 && observed.input_ids == case.input_ids {
            gates.coverage_8192 = true;
        }
        if model.architecture.family == Family::Qwen3 && model.operation == Operation::Rerank {
            if observed.input_ids != case.input_ids {
                gates.semantics = false;
                result.failures.push("qwen_template_mismatch".into());
            }
            let readout = &model.grammar.readout;
            let expected = readout
                .yes
                .as_ref()
                .zip(readout.no.as_ref())
                .map(|(yes, no)| (yes.id, no.id));
            if observed.readout != expected || expected.is_none() {
                gates.semantics = false;
                result.failures.push("qwen_readout_mismatch".into());
            }
        }
        match (&case.output, &observed.output) {
            (Output::Embedding(reference), Output::Embedding(actual)) => {
                let finite = actual.iter().all(|x| x.is_finite());
                gates.dimension &= actual.len() == model.output.dimension as usize;
                gates.finite &= finite;
                if !finite || actual.len() != reference.len() || norm(actual) == 0.0 {
                    gates.unit_norm = false;
                    gates.cosine = false;
                    continue;
                }
                let error = (norm(actual) - 1.0).abs();
                let cosine = actual
                    .iter()
                    .zip(reference)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    / (norm(actual) * norm(reference));
                max_norm = max_norm.max(error);
                min_cosine = min_cosine.min(cosine);
                gates.unit_norm &= error <= 1e-3;
                gates.cosine &= cosine.is_finite() && cosine >= 0.999;
            }
            (Output::Score(reference), Output::Score(actual)) => {
                gates.finite &= actual.is_finite();
                if !actual.is_finite() {
                    gates.score = false;
                    gates.ranking = false;
                    continue;
                }
                max_error = max_error.max((actual - reference).abs());
                if let Some(pool) = &case.pool {
                    let (refs, scores) = pools.entry(pool.clone()).or_default();
                    refs.push(*reference);
                    scores.push(*actual);
                }
            }
            _ => {
                gates.dimension = false;
                gates.finite = false;
                gates.score = false;
                gates.ranking = false;
                gates.unit_norm = false;
                gates.cosine = false;
            }
        }
    }
    if model.operation == Operation::Embed {
        result.metrics.min_cosine = Some(min_cosine);
        result.metrics.max_norm_error = Some(max_norm);
    } else {
        result.metrics.max_absolute_error = Some(max_error);
        gates.score &= max_error
            <= match profile.rerank_tolerance_class {
                ToleranceClass::Fp32 => 0.005,
                ToleranceClass::Fp16 => 0.02,
            };
        for (id, (refs, scores)) in pools {
            let metric = ranking(&refs, &scores)?;
            gates.ranking &= metric.passed;
            result.metrics.pools.insert(id, metric);
        }
        gates.ranking &= !result.metrics.pools.is_empty();
    }
    gates.coverage = ["short", "shape_boundary", "batched", "long"]
        .iter()
        .all(|c| categories.contains(c))
        && (model.operation == Operation::Embed
            || (categories.contains("pool_10") && categories.contains("pool_100")));
    if !gates.complete {
        gates.dimension = false;
        gates.finite = false;
        gates.unit_norm = false;
        gates.cosine = false;
        gates.score = false;
        gates.ranking = false;
        gates.coverage = false;
        gates.coverage_8192 = false;
        gates.semantics = false;
        result.failures.push("incomplete_output".into());
    }
    result.failures.sort();
    result.failures.dedup();
    result.gates = gates;
    Ok(result)
}

/// Group consecutive candidates by their final model-input token counts,
/// including template and special tokens, rather than admission estimates.
/// A candidate with 8192 tokens gets its own call even if the total budget is larger.
pub fn split_pool(
    tokens: &[usize],
    max_items: usize,
    token_budget: usize,
) -> Result<Vec<Range<usize>>> {
    if max_items == 0 || token_budget == 0 {
        return Err(perr!("invalid inline budget"));
    }
    if tokens
        .iter()
        .any(|&n| n == 0 || n > 8192 || n > token_budget)
    {
        return Err(perr!("candidate exceeds composed token budget"));
    }
    let mut calls = vec![];
    let mut start = 0;
    while start < tokens.len() {
        let mut end = start;
        let mut used = 0;
        while end < tokens.len() && end - start < max_items {
            let n = tokens[end];
            if n > token_budget - used || (end > start && (n == 8192 || tokens[start] == 8192)) {
                break;
            }
            used += n;
            end += 1;
        }
        calls.push(start..end);
        start = end;
    }
    Ok(calls)
}

/// The callback should issue one module `rerank.score` request for the range
/// supplied, returning API-domain scores in request order.
pub fn score_pool(
    tokens: &[usize],
    max_items: usize,
    token_budget: usize,
    mut call: impl FnMut(Range<usize>) -> Result<Vec<f64>>,
) -> Result<Vec<f64>> {
    let mut scores = Vec::with_capacity(tokens.len());
    for range in split_pool(tokens, max_items, token_budget)? {
        let output = call(range.clone())?;
        if output.len() != range.len() || output.iter().any(|x| !x.is_finite()) {
            return Err(perr!(
                "invalid rerank.score response for {}..{}",
                range.start,
                range.end
            ));
        }
        scores.extend(output);
    }
    Ok(scores)
}

/// A load-time subset checks numerical and semantic gates, not full-suite
/// coverage. The underlying comparison and manifest tolerances are identical
/// to certification; this result cannot certify 8192-token or corpus coverage.
#[derive(Debug, Serialize)]
pub struct SubsetEvaluation {
    pub evaluation: Evaluation,
}
impl SubsetEvaluation {
    pub fn passed(&self) -> bool {
        let gates = &self.evaluation.gates;
        gates.complete
            && gates.dimension
            && gates.finite
            && gates.unit_norm
            && gates.cosine
            && gates.score
            && gates.ranking
            && gates.semantics
    }
}

pub fn evaluate_subset(
    manifest: &Manifest,
    profile_id: &str,
    fingerprint: &str,
    fixtures: &FixtureSet,
    outputs: &BTreeMap<String, ObservedCase>,
) -> Result<SubsetEvaluation> {
    evaluate(manifest, profile_id, fingerprint, fixtures, outputs)
        .map(|evaluation| SubsetEvaluation { evaluation })
}
