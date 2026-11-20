#![cfg(feature = "vulkan")]

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};
use synapse_parity::manifest::{Family, Manifest, Operation, MODEL_SLUGS};
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

/// Cosine between one output vector and its fp32 reference, after checking
/// the vector is unit length.
fn checked_cosine(slug: &str, case: &Case, vector: &[f32]) -> f64 {
    let expected: Vec<f64> = serde_json::from_value(case.output.clone()).unwrap();
    assert_eq!(expected.len(), vector.len());
    let norm = vector
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() <= 1e-3, "{slug} {} norm {norm}", case.id);
    let reference_norm = expected.iter().map(|v| v * v).sum::<f64>().sqrt();
    vector
        .iter()
        .zip(&expected)
        .map(|(a, b)| f64::from(*a) * b)
        .sum::<f64>()
        / (norm * reference_norm)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn read_fixture(repo: &Path, slug: &str) -> Fixture {
    let fixture: Fixture = serde_json::from_slice(
        &std::fs::read(repo.join(format!(
            "bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json"
        )))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(fixture.model, slug);
    fixture
}

fn load_engine(manifest: &Manifest, packages: &Path, slug: &str) -> Engine {
    let model = manifest.models[slug].clone();
    let profile_name = format!("{slug}.owned-vulkan");
    let profile = manifest.profiles[&profile_name].clone();
    let required = requirements(manifest, Some(slug)).unwrap();
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
    let package = packages.join(slug);
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
    synapse_parity::safetensors::verify_package_structure(&bytes, &profile_name).unwrap();
    let (header, data) = synapse_parity::safetensors::split_file(&bytes).unwrap();
    let header = synapse_parity::safetensors::parse_header_json(header).unwrap();
    Engine::load(model, profile, prepared, required, &header, data).unwrap()
}

#[test]
#[ignore = "requires GPU, converted SYNAPSE_VULKAN_TEST_PACKAGES; macOS additionally requires moltenvk-diagnostic and SYNAPSE_MOLTENVK_LOADER"]
fn four_models_match_reference_vectors_scores_window_and_padding() {
    let repo = repo_root();
    let root = PathBuf::from(
        std::env::var_os("SYNAPSE_VULKAN_TEST_PACKAGES").expect("converted package root"),
    );
    assert!(root.canonicalize().unwrap().starts_with(&repo));
    let manifest = synapse_worker_vulkan::manifest();
    let mut outputs: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for slug in MODEL_SLUGS {
        let model = &manifest.models[slug];
        let fixture = read_fixture(&repo, slug);
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
        let engine = load_engine(&manifest, &root, slug);
        let actual = engine.infer(&sequences).unwrap();
        assert!(actual.iter().all(|v| v.is_finite()));
        if model.operation == Operation::Embed {
            let dim = model.output().unwrap().dimension as usize;
            assert_eq!(actual.len(), selected.len() * dim);
            let mut min_cosine = 1.0f64;
            for (row, case) in selected.iter().enumerate() {
                let cosine = checked_cosine(slug, case, &actual[row * dim..(row + 1) * dim]);
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
        for (row, case) in selected.iter().enumerate() {
            let width = actual.len() / selected.len();
            outputs.insert(
                format!("{slug}/{}", case.id),
                actual[row * width..(row + 1) * width].to_vec(),
            );
        }
        // The remaining length cases, up to the 8192-token row, run one per
        // request so that each is padded only to its own length.
        let lengths: Vec<&Case> = fixture
            .cases
            .iter()
            .filter(|c| c.category == "shape_boundary" || c.category == "long")
            .filter(|c| !selected.iter().any(|s| s.id == c.id))
            .collect();
        assert!(
            lengths.iter().any(|c| c.input_ids.len() == 8192),
            "must exercise the 8192-token context"
        );
        for case in lengths {
            let actual = engine.infer(std::slice::from_ref(&case.input_ids)).unwrap();
            assert!(actual.iter().all(|v| v.is_finite()));
            let tokens = case.input_ids.len();
            if model.operation == Operation::Embed {
                let cosine = checked_cosine(slug, case, &actual);
                println!(
                    "GPU_PARITY {slug} {} tokens={tokens} cosine={cosine:.9}",
                    case.id
                );
                assert!(cosine >= 0.999, "{slug} {} cosine {cosine}", case.id);
            } else {
                assert_eq!(actual.len(), 1);
                let error = (f64::from(actual[0]) - case.output.as_f64().unwrap()).abs();
                println!(
                    "GPU_PARITY {slug} {} tokens={tokens} score_error={error:.9}",
                    case.id
                );
                assert!(error <= 0.02, "{slug} {} score error {error}", case.id);
            }
            outputs.insert(format!("{slug}/{}", case.id), actual);
        }
    }
    // Optional dump of every output, to compare two kernel revisions directly.
    if let Some(path) = std::env::var_os("SYNAPSE_VULKAN_PARITY_OUTPUTS") {
        std::fs::write(path, serde_json::to_vec(&outputs).unwrap()).unwrap();
    }
}

fn check_row_bits(
    engine: &Engine,
    slug: &str,
    dim: usize,
    rows: &[Vec<i32>],
    output_dir: Option<&Path>,
) -> std::time::Duration {
    let width = rows.iter().map(Vec::len).max().unwrap();
    let start = Instant::now();
    let batch = engine.infer(rows).unwrap();
    let elapsed = start.elapsed();
    assert_eq!(batch.len(), rows.len() * dim, "{slug} output shape");
    for (row, actual) in rows.iter().zip(batch.chunks_exact(dim)) {
        let len = row.len();
        let alone = engine.infer(std::slice::from_ref(row)).unwrap();
        assert_eq!(alone.len(), dim, "{slug} singleton output shape");
        let mut differing_values = 0;
        let mut differing_bits = 0;
        let mut max_abs_diff = 0.0f32;
        for (&a, &b) in actual.iter().zip(&alone) {
            assert!(a.is_finite() && b.is_finite(), "{slug} {len}/{width}");
            let diff = a.to_bits() ^ b.to_bits();
            differing_values += usize::from(diff != 0);
            differing_bits += diff.count_ones();
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        assert!(
            differing_values == 0,
            "{slug}: {len}-token row in a {width}-token batch differs from singleton in \
             {differing_values}/{dim} values ({differing_bits} bits, max abs diff {max_abs_diff:e})"
        );
        println!(
            "ROW_WIDTH_IDENTITY {slug} row={len} batch_max={width} values={dim} bit-identical"
        );
        if let Some(dir) = output_dir {
            let bytes: Vec<_> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{slug}-{len}-of-{width}.f32le")), bytes).unwrap();
        }
    }
    elapsed
}

#[test]
fn mixed_length_rows_match_singletons_bit_for_bit() {
    let Some(packages) = std::env::var_os("SYNAPSE_VULKAN_TEST_PACKAGES") else {
        eprintln!("SKIP mixed_length_rows_match_singletons_bit_for_bit: SYNAPSE_VULKAN_TEST_PACKAGES is unset");
        return;
    };
    let packages = PathBuf::from(packages);
    let output_dir = std::env::var_os("SYNAPSE_VULKAN_ROW_WIDTH_OUTPUTS").map(PathBuf::from);
    let manifest = synapse_worker_vulkan::manifest();
    let repo = repo_root();
    for slug in MODEL_SLUGS {
        let model = &manifest.models[slug];
        let dim = model.output().unwrap().dimension as usize;
        let fixture = read_fixture(&repo, slug);
        let tokens = |id| &fixture.cases.iter().find(|c| c.id == id).unwrap().input_ids;
        let rows: Vec<_> = if model.architecture.family == Family::Modernbert {
            [7, 300, 1_000]
                .map(|len| tokens("long-8192")[..len].to_vec())
                .into()
        } else {
            ["boundary-129", "boundary-513"]
                .map(|id| tokens(id).clone())
                .into()
        };
        let engine = load_engine(&manifest, &packages, slug);
        check_row_bits(&engine, slug, dim, &rows, output_dir.as_deref());
        // Keep the longer timing case opt-in; software Vulkan is much slower.
        if output_dir.is_some() && slug == "gte-reranker-modernbert-base" {
            let rows = [
                tokens("short-0").clone(),
                tokens("long-8192")[..4_096].to_vec(),
            ];
            assert_eq!(rows[0].len(), 32);
            let elapsed = check_row_bits(&engine, slug, dim, &rows, output_dir.as_deref());
            println!(
                "ROW_WIDTH_TIMING {slug} 32+4096 {:.3} ms",
                elapsed.as_secs_f64() * 1_000.0
            );
        }
    }
}
