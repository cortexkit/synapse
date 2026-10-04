use std::collections::BTreeMap;

use synapse_parity::canonical::sha256_hex;
use synapse_parity::evaluator::{
    evaluate, ranking, score_pool, split_pool, FixtureSet, ObservedCase, Output,
};
use synapse_parity::manifest::{Manifest, Operation, ToleranceClass};
use synapse_parity::parity_dir;

fn manifest() -> Manifest {
    Manifest::load(&parity_dir().join("models.json")).unwrap()
}

fn reference_outputs(
    manifest: &Manifest,
    model: &str,
    fixtures: &FixtureSet,
) -> BTreeMap<String, ObservedCase> {
    let readout = &manifest.models[model].grammar.readout;
    let readout = readout
        .yes
        .as_ref()
        .zip(readout.no.as_ref())
        .map(|(yes, no)| (yes.id, no.id));
    fixtures
        .cases()
        .iter()
        .map(|c| {
            (
                c.id.clone(),
                ObservedCase {
                    output: c.output.clone(),
                    input_ids: c.input_ids.clone(),
                    readout,
                },
            )
        })
        .collect()
}

#[test]
fn committed_references_pass_every_profile_and_report_identity() {
    let manifest = manifest();
    for (id, profile) in &manifest.profiles {
        let fixtures = FixtureSet::load(&parity_dir(), &manifest, &profile.model).unwrap();
        let outputs = reference_outputs(&manifest, &profile.model, &fixtures);
        let report = evaluate(&manifest, id, "runtime-fingerprint", &fixtures, &outputs).unwrap();
        assert!(
            report.gates.complete
                && report.gates.finite
                && report.gates.coverage
                && report.gates.coverage_8192
                && report.gates.semantics,
            "{id}: {report:?}"
        );
        match report.operation {
            Operation::Embed => {
                assert!(report.gates.dimension && report.gates.unit_norm && report.gates.cosine)
            }
            Operation::Rerank => {
                assert!(
                    report.gates.score && report.gates.ranking,
                    "{id}: {report:?}"
                );
                assert_eq!(report.tolerance_class, profile.rerank_tolerance_class);
                assert_eq!(report.metrics.pools.len(), 2);
            }
        }
        assert_eq!(report.completed_fixtures, fixtures.cases().len());
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["profile_id"], *id);
        assert_eq!(json["model"], profile.model);
        assert_eq!(json["fingerprint"], "runtime-fingerprint");
        assert!(json["fixture_set_id"]
            .as_str()
            .unwrap()
            .contains("transformers-5.16.1.seed-0"));
    }
}

#[test]
fn no_output_fails_every_gate_with_zero_completed_fixtures() {
    let manifest = manifest();
    for (id, profile) in &manifest.profiles {
        let fixtures = FixtureSet::load(&parity_dir(), &manifest, &profile.model).unwrap();
        let report = evaluate(&manifest, id, "missing", &fixtures, &BTreeMap::new()).unwrap();
        assert_eq!(report.completed_fixtures, 0);
        let gates = serde_json::to_value(report.gates).unwrap();
        assert!(gates.as_object().unwrap().values().all(|v| v == false));
    }
}

#[test]
fn partial_output_fails_every_gate_and_reports_incomplete_output() {
    let manifest = manifest();
    for profile_id in [
        "gte-modernbert-base.owned-metal",
        "gte-reranker-modernbert-base.owned-metal",
    ] {
        let model = &manifest.profiles[profile_id].model;
        let fixtures = FixtureSet::load(&parity_dir(), &manifest, model).unwrap();
        let mut outputs = reference_outputs(&manifest, model, &fixtures);
        let complete = evaluate(&manifest, profile_id, "fp", &fixtures, &outputs).unwrap();
        let gates = serde_json::to_value(&complete.gates).unwrap();
        assert!(
            gates.as_object().unwrap().values().all(|v| v == true),
            "{profile_id}: {complete:?}"
        );
        let expected = fixtures.cases().len();
        assert_eq!(complete.expected_fixtures, expected);
        assert_eq!(complete.completed_fixtures, expected);

        assert!(outputs.remove(&fixtures.cases()[0].id).is_some());
        assert_eq!(outputs.len(), expected - 1);
        assert!(!outputs.is_empty());
        let partial = evaluate(&manifest, profile_id, "fp", &fixtures, &outputs).unwrap();
        assert_eq!(partial.expected_fixtures, expected);
        assert_eq!(partial.completed_fixtures, expected - 1);
        let gates = serde_json::to_value(&partial.gates).unwrap();
        assert!(
            gates.as_object().unwrap().values().all(|v| v == false),
            "{profile_id}: {partial:?}"
        );
        assert!(partial
            .failures
            .iter()
            .any(|failure| failure == "incomplete_output"));
    }
}

#[test]
fn embedding_dimension_finiteness_norm_and_cosine_are_independent() {
    let manifest = manifest();
    let model = "gte-modernbert-base";
    let fixtures = FixtureSet::load(&parity_dir(), &manifest, model).unwrap();
    let id = &fixtures.cases()[0].id;
    for kind in ["dimension", "finite", "norm", "cosine"] {
        let mut outputs = reference_outputs(&manifest, model, &fixtures);
        let Output::Embedding(v) = &mut outputs.get_mut(id).unwrap().output else {
            panic!()
        };
        match kind {
            "dimension" => {
                v.pop();
            }
            "finite" => v[0] = f64::NAN,
            "norm" => v.iter_mut().for_each(|x| *x *= 2.0),
            "cosine" => v.iter_mut().for_each(|x| *x = -*x),
            _ => unreachable!(),
        }
        let report = evaluate(
            &manifest,
            "gte-modernbert-base.owned-metal",
            "fp",
            &fixtures,
            &outputs,
        )
        .unwrap();
        match kind {
            "dimension" => assert!(!report.gates.dimension),
            "finite" => assert!(!report.gates.finite),
            "norm" => assert!(!report.gates.unit_norm && report.gates.cosine),
            "cosine" => assert!(!report.gates.cosine && report.gates.unit_norm),
            _ => unreachable!(),
        }
    }
}

#[test]
fn every_rerank_profile_uses_its_manifest_tolerance_in_api_domain() {
    let manifest = manifest();
    for (id, profile) in &manifest.profiles {
        if manifest.models[&profile.model].operation != Operation::Rerank {
            continue;
        }
        let fixtures = FixtureSet::load(&parity_dir(), &manifest, &profile.model).unwrap();
        let tolerance = match profile.rerank_tolerance_class {
            ToleranceClass::Fp32 => 0.005,
            ToleranceClass::Fp16 => 0.02,
        };
        for (delta, expected) in [(tolerance * 0.99, true), (tolerance * 1.01, false)] {
            let mut outputs = reference_outputs(&manifest, &profile.model, &fixtures);
            let Output::Score(s) = &mut outputs.get_mut(&fixtures.cases()[0].id).unwrap().output
            else {
                panic!()
            };
            *s += delta;
            let report = evaluate(&manifest, id, "fp", &fixtures, &outputs).unwrap();
            assert_eq!(report.gates.score, expected, "{id} delta {delta}");
            assert_eq!(report.tolerance_class, profile.rerank_tolerance_class);
        }
    }
}

#[test]
fn near_tie_chain_is_not_transitive() {
    let reference: Vec<f64> = (0..10).map(|i| i as f64 * 0.002).collect();
    assert!(reference.windows(2).all(|p| p[1] - p[0] < 0.01));
    let mut actual = reference.clone();
    actual.swap(0, 9);
    let report = ranking(&reference, &actual).unwrap();
    assert!(!report.top10 && !report.passed);
}

#[test]
fn gap_filtered_tau_counts_ties_and_ignores_near_pairs() {
    for reference in [vec![0.5, 0.5, 0.505], vec![0.9, 0.5, 0.505, 0.1]] {
        let identical = ranking(&reference, &reference).unwrap();
        assert_eq!(identical.tau, 1.0);
        assert!(identical.passed);
        let mut actual = reference.clone();
        actual.swap(1, 2);
        let swapped = ranking(&reference, &actual).unwrap();
        assert_eq!(identical.tau, swapped.tau);
        assert_eq!(identical.passed, swapped.passed);
    }
    let tied = ranking(&[0.9, 0.5, 0.1], &[0.8, 0.8, 0.1]).unwrap();
    assert_eq!(
        (tied.concordant, tied.discordant, tied.backend_ties),
        (2, 0, 1)
    );
    assert_eq!(tied.tau, 2.0 / 3.0);
    assert!(!tied.passed);
    assert!(ranking(&[0.1], &[0.9]).unwrap().passed);
    assert!(ranking(&[0.1, 0.2], &[f64::NAN, 0.2]).is_err());
}

#[test]
fn hundred_candidate_ranks_9_to_12_match_recorded_tau_and_pairwise_gate() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/ranks-9-to-12.json")).unwrap();
    let reference: Vec<f64> = (0..100).map(|i| (100 - i) as f64 * 0.02).collect();
    let mut actual = reference.clone();
    for (offset, source) in fixture["permutation"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        actual[8 + offset] = reference[source.as_u64().unwrap() as usize - 1];
    }
    let report = ranking(&reference, &actual).unwrap();
    assert_eq!(report.concordant, 4944);
    assert_eq!(report.discordant, 6);
    assert!((report.tau - fixture["expected_tau"].as_f64().unwrap()).abs() < 1e-12);
    assert_eq!(report.top10, fixture["expected_top10"].as_bool().unwrap());
    assert_eq!(report.passed, fixture["expected_pass"].as_bool().unwrap());
    // Six inverted pairs barely affect tau across 4950 pairs, but reversing
    // ranks 9–12 places reference top-10 candidates below lower-scoring ones.
    assert!(report.tau >= 0.99);
}

#[test]
fn pool_calls_are_consecutive_bounded_and_reassembled_in_candidate_order() {
    let tokens: Vec<usize> = (0..100).map(|i| 30 + i % 7).collect();
    let mut next = 0;
    let scores = score_pool(&tokens, 16, 300, |range| {
        assert_eq!(range.start, next);
        assert!(range.len() <= 16 && tokens[range.clone()].iter().sum::<usize>() <= 300);
        next = range.end;
        Ok(range.map(|i| i as f64 / 100.0).collect())
    })
    .unwrap();
    assert_eq!(next, 100);
    assert_eq!(
        scores,
        (0..100).map(|i| i as f64 / 100.0).collect::<Vec<_>>()
    );
    assert_eq!(
        split_pool(&[2, 8192, 3], 100, 16384).unwrap(),
        vec![0..1, 1..2, 2..3]
    );
    assert!(split_pool(&[8193], 16, 16384).is_err());
    assert!(split_pool(&[8192], 16, 8191).is_err());
    assert!(split_pool(&[1], 0, 1).is_err());
    assert!(score_pool(&[1, 1], 2, 2, |_| Ok(vec![0.1])).is_err());
}

#[test]
fn fixture_id_and_digest_mismatches_are_refused() {
    let manifest = manifest();
    let model = "gte-modernbert-base";
    let bytes = std::fs::read(parity_dir().join(format!(
        "fixtures/{model}/{model}.ref-v1.transformers-5.16.1.seed-0.json"
    )))
    .unwrap();
    assert!(
        FixtureSet::from_bytes(&manifest, model, &bytes, "wrong digest")
            .unwrap_err()
            .0
            .contains("fixture_digest_mismatch")
    );
    let mut document: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    document["fixture_set_id"] = "wrong-id".into();
    let wrong = serde_json::to_vec(&document).unwrap();
    assert!(
        FixtureSet::from_bytes(&manifest, model, &wrong, &sha256_hex(&wrong))
            .unwrap_err()
            .0
            .contains("fixture_set_mismatch")
    );
    let fixtures = FixtureSet::load(&parity_dir(), &manifest, model).unwrap();
    assert!(evaluate(
        &manifest,
        "qwen3-embedding-0.6b.owned-metal",
        "fp",
        &fixtures,
        &BTreeMap::new()
    )
    .is_err());
}

#[test]
fn qwen_template_mismatch_is_named() {
    let manifest = manifest();
    let model = "qwen3-reranker-0.6b";
    let fixtures = FixtureSet::load(&parity_dir(), &manifest, model).unwrap();
    let mut outputs = reference_outputs(&manifest, model, &fixtures);
    outputs
        .get_mut(&fixtures.cases()[0].id)
        .unwrap()
        .input_ids
        .pop();
    let report = evaluate(
        &manifest,
        "qwen3-reranker-0.6b.owned-metal",
        "fp",
        &fixtures,
        &outputs,
    )
    .unwrap();
    assert!(!report.gates.semantics);
    assert_eq!(report.failures, ["qwen_template_mismatch"]);
}

#[test]
fn qwen_readout_mismatch_is_named() {
    let manifest = manifest();
    let model = "qwen3-reranker-0.6b";
    let fixtures = FixtureSet::load(&parity_dir(), &manifest, model).unwrap();
    let mut outputs = reference_outputs(&manifest, model, &fixtures);
    let observed = outputs.get_mut(&fixtures.cases()[0].id).unwrap();
    observed.readout = observed.readout.map(|(yes, no)| (no, yes));
    let report = evaluate(
        &manifest,
        "qwen3-reranker-0.6b.owned-metal",
        "fp",
        &fixtures,
        &outputs,
    )
    .unwrap();
    assert!(!report.gates.semantics);
    assert_eq!(report.failures, ["qwen_readout_mismatch"]);
}
