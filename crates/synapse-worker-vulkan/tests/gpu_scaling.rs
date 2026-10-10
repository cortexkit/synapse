#![cfg(feature = "vulkan")]

//! Development-only timing of one sequence as its length doubles from 512 to
//! 8192 tokens. Attention cost grows with the square of the length while every
//! other stage grows linearly, so the ratio between successive rows shows which
//! of the two dominates at long context. Absolute times depend on the GPU.

use serde::Deserialize;
use std::{path::PathBuf, time::Instant};
use synapse_worker_vulkan::{admission::requirements, runtime::Engine};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    input_ids: Vec<i32>,
}

#[test]
#[ignore = "requires GPU, converted SYNAPSE_VULKAN_TEST_PACKAGES; macOS additionally requires moltenvk-diagnostic and SYNAPSE_MOLTENVK_LOADER"]
fn one_row_time_from_512_to_8192_tokens() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let root = PathBuf::from(
        std::env::var_os("SYNAPSE_VULKAN_TEST_PACKAGES").expect("converted package root"),
    );
    // Optional comma-separated model slugs; both families by default.
    let slugs = std::env::var("SYNAPSE_VULKAN_SCALING_MODELS")
        .unwrap_or_else(|_| "qwen3-embedding-0.6b,gte-modernbert-base".into());
    let manifest = synapse_worker_vulkan::manifest();
    for slug in slugs.split(',') {
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
        let bytes = std::fs::read(root.join(slug).join("model.safetensors")).unwrap();
        let (header, data) = synapse_parity::safetensors::split_file(&bytes).unwrap();
        let header = synapse_parity::safetensors::parse_header_json(header).unwrap();
        let fixture: Fixture = serde_json::from_slice(
            &std::fs::read(repo.join(format!(
                "bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json"
            )))
            .unwrap(),
        )
        .unwrap();
        let long = fixture
            .cases
            .iter()
            .find(|c| c.id == "long-8192")
            .expect("8192-token fixture case");
        let engine = Engine::load(model, profile, prepared, required, &header, data).unwrap();
        drop(bytes);
        // Warm the pipeline and the arena once before timing.
        engine.infer(&[long.input_ids[..512].to_vec()]).unwrap();
        let mut previous: Option<f64> = None;
        for length in [512, 1024, 2048, 4096, 8192] {
            let row = long.input_ids[..length].to_vec();
            let start = Instant::now();
            let output = engine.infer(&[row]).unwrap();
            let seconds = start.elapsed().as_secs_f64();
            assert!(output.iter().all(|v| v.is_finite()));
            println!(
                "GPU_SCALING {slug} tokens={length} seconds={seconds:.3} growth={}",
                previous.map_or("-".into(), |p| format!("{:.2}", seconds / p))
            );
            previous = Some(seconds);
        }
    }
}
