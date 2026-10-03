#![cfg(feature = "vulkan")]

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};
use synapse_parity::manifest::{Operation, MODEL_SLUGS};
use synapse_worker_vulkan::{admission::requirements, runtime::Engine};

#[derive(Deserialize)]
struct Fixture {
    model: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    category: String,
    input_ids: Vec<i32>,
    output: serde_json::Value,
    pool: Option<String>,
}

fn rank_gate(expected: &[f64], actual: &[f32]) -> f64 {
    let mut ordered: Vec<usize> = (0..expected.len()).collect();
    ordered.sort_by(|&a, &b| expected[b].total_cmp(&expected[a]));
    let mut comparable = 0;
    let mut discordant = 0;
    for i in 0..expected.len() {
        for j in i + 1..expected.len() {
            if (expected[i] - expected[j]).abs() < 0.01 {
                continue;
            }
            comparable += 1;
            if (expected[i] - expected[j]).signum() != f64::from(actual[i] - actual[j]).signum() {
                discordant += 1;
            }
        }
    }
    for &i in ordered.iter().take(10) {
        for j in 0..expected.len() {
            if expected[i] - expected[j] >= 0.01 {
                assert!(
                    actual[i] > actual[j],
                    "reference top order reversed at {i}/{j}"
                );
            }
        }
    }
    let tau = if comparable == 0 {
        1.0
    } else {
        1.0 - 2.0 * f64::from(discordant) / f64::from(comparable)
    };
    assert!(tau >= 0.99, "Kendall tau {tau}");
    tau
}

#[test]
#[ignore = "requires GPU, converted SYNAPSE_VULKAN_TEST_PACKAGES; macOS additionally requires moltenvk-diagnostic and SYNAPSE_MOLTENVK_LOADER"]
fn four_models_match_reference_vectors_scores_window_and_padding() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let root = PathBuf::from(
        std::env::var_os("SYNAPSE_VULKAN_TEST_PACKAGES").expect("converted package root"),
    );
    assert!(root.canonicalize().unwrap().starts_with(&repo));
    let manifest = synapse_worker_vulkan::manifest();
    for slug in MODEL_SLUGS {
        let model = manifest.models[slug].clone();
        let profile = manifest.profiles[&format!("{slug}.owned-vulkan")].clone();
        let required = requirements(&manifest, Some(slug)).unwrap();
        #[cfg(all(target_os = "macos", feature = "moltenvk-diagnostic"))]
        let prepared = synapse_worker_vulkan::runtime::prepare_moltenvk_parity(
            &PathBuf::from(
                std::env::var_os("SYNAPSE_MOLTENVK_LOADER").expect("actual SDK loader dylib"),
            ),
            required,
        )
        .unwrap();
        #[cfg(not(all(target_os = "macos", feature = "moltenvk-diagnostic")))]
        let prepared = synapse_worker_vulkan::runtime::prepare(required).unwrap();
        let package = root.join(slug);
        assert!(!package.join("config.json").exists());
        let bytes = std::fs::read(package.join("model.safetensors")).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            profile
                .converted_package_digest
                .as_deref()
                .unwrap()
                .trim_start_matches("sha256:")
        );
        synapse_parity::safetensors::verify_package_structure(
            &bytes,
            &format!("{slug}.owned-vulkan"),
        )
        .unwrap();
        let (header, data) = synapse_parity::safetensors::split_file(&bytes).unwrap();
        let header = synapse_parity::safetensors::parse_header_json(header).unwrap();
        let fixture: Fixture = serde_json::from_slice(
            &std::fs::read(repo.join(format!(
                "bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json"
            )))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(fixture.model, slug);
        let selected: Vec<&Case> = fixture
            .cases
            .iter()
            .filter(|c| {
                c.category == "short"
                    || c.category == "batched"
                    || c.id == "boundary-129"
                    || c.pool.as_deref() == Some("pool-10")
            })
            .collect();
        assert!(
            selected.iter().any(|c| c.input_ids.len() > 128),
            "must exercise local window"
        );
        assert!(
            selected
                .iter()
                .any(|c| c.input_ids.len() != selected[0].input_ids.len()),
            "must exercise padding"
        );
        let sequences: Vec<_> = selected.iter().map(|c| c.input_ids.clone()).collect();
        let engine =
            Engine::load(model.clone(), profile, prepared, required, &header, data).unwrap();
        drop(bytes);
        let actual = engine.infer(&sequences).unwrap();
        assert!(actual.iter().all(|v| v.is_finite()));
        if model.operation == Operation::Embed {
            let dim = model.output.dimension as usize;
            assert_eq!(actual.len(), selected.len() * dim);
            let mut min_cosine = 1.0f64;
            for (row, case) in selected.iter().enumerate() {
                let vector = &actual[row * dim..(row + 1) * dim];
                let expected: Vec<f64> = serde_json::from_value(case.output.clone()).unwrap();
                assert_eq!(expected.len(), dim);
                let norm = vector
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!((norm - 1.0).abs() <= 1e-3, "{slug} {} norm {norm}", case.id);
                let reference_norm = expected.iter().map(|v| v * v).sum::<f64>().sqrt();
                let cosine = vector
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| f64::from(*a) * b)
                    .sum::<f64>()
                    / (norm * reference_norm);
                min_cosine = min_cosine.min(cosine);
                eprintln!("{slug} {} cosine={cosine:.9}", case.id);
            }
            println!(
                "GPU_PARITY {slug} min_cosine={min_cosine:.9} rows={} padded_width={}",
                selected.len(),
                sequences.iter().map(Vec::len).max().unwrap()
            );
            assert!(min_cosine >= 0.999, "{slug} min cosine {min_cosine}");
        } else {
            assert_eq!(actual.len(), selected.len());
            let expected: Vec<_> = selected
                .iter()
                .map(|c| c.output.as_f64().unwrap())
                .collect();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0.0f64, f64::max);
            let mut pools: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
            for (index, case) in selected.iter().enumerate() {
                if let Some(pool) = &case.pool {
                    pools.entry(pool).or_default().push(index);
                }
            }
            assert!(!pools.is_empty(), "reranker must exercise a candidate pool");
            let mut min_tau = 1.0f64;
            for indices in pools.values() {
                min_tau = min_tau.min(rank_gate(
                    &indices.iter().map(|&i| expected[i]).collect::<Vec<_>>(),
                    &indices.iter().map(|&i| actual[i]).collect::<Vec<_>>(),
                ));
            }
            println!("GPU_PARITY {slug} max_score_error={max_error:.9} min_kendall_tau={min_tau:.9} rows={}", selected.len());
            assert!(max_error <= 0.02, "{slug} max score error {max_error}");
        }
    }
}
