//! Real-NVIDIA check of the owned-CUDA shape-plan cache.
//!
//! The ModernBERT and Qwen3 engines each keep compiled plans (an arena sized
//! for one exact batch x sequence shape, plus its CUDA graph) for at most two
//! shapes. These tests drive a loaded model through two cycles of five
//! distinct sequence lengths and read device memory in-process with
//! `cudaMemGetInfo` after each cycle. The second cycle uses five lengths not
//! seen in the first, so an unbounded cache would keep five more arenas
//! resident and used memory would rise by far more than the slack allowed.
//!
//! Every length is embedded twice in a row: the first call misses the cache
//! and is served from the plan's first forward, the second hits the cached
//! plan; both must return bit-identical vectors. Each model runs once with
//! CUDA graph replay on and once with it off, and the two runs must also
//! agree bit for bit.
//!
//! Ignored by default. Run them on an NVIDIA machine with a build that has the
//! `cuda` feature, pointing each variable at a converted owned-cuda package
//! (the directory holding `model.safetensors`, or the file itself):
//!
//! ```sh
//! SYNAPSE_CUDA_PLAN_MODERNBERT_PACKAGE=/abs/gte-modernbert-base \
//! SYNAPSE_CUDA_PLAN_QWEN3_PACKAGE=/abs/qwen3-embedding-0.6b \
//! cargo test --locked --release -p synapse-engine-cuda --features cuda \
//!   --test shape_plan_hardware -- --ignored --nocapture --test-threads=1
//! ```

use synapse_core::{EmbedEngine, LoadedModel, RuntimeConfig, ValidatedArtifact};
use synapse_engine_cuda::manifest::Profile;
use synapse_engine_cuda::{ModelFamily, OwnedCudaEmbedEngine};

/// Sequence lengths of the first cycle. The second cycle adds one token to
/// each, so every shape in it is new to the engine.
const LENGTHS: [usize; 5] = [512, 1024, 2048, 3072, 4096];

/// Allowed rise in device memory from the end of cycle one to the end of
/// cycle two. Both cycles end with the plans for their two longest lengths
/// resident, which differ by one token, so the only expected variation is
/// allocator granularity and cuBLASLt workspace choice. Five extra retained
/// ModernBERT plans at these lengths would add well over a gigabyte.
const SLACK_BYTES: u64 = 256 << 20;

#[test]
#[ignore = "needs an NVIDIA GPU, --features cuda and SYNAPSE_CUDA_PLAN_MODERNBERT_PACKAGE"]
fn modernbert_shape_plans_stay_bounded_and_bit_identical() {
    check_family(
        "gte-modernbert-base.owned-cuda",
        ModelFamily::GteModernBert,
        "SYNAPSE_CUDA_PLAN_MODERNBERT_PACKAGE",
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU, --features cuda and SYNAPSE_CUDA_PLAN_QWEN3_PACKAGE"]
fn qwen3_shape_plans_stay_bounded_and_bit_identical() {
    check_family(
        "qwen3-embedding-0.6b.owned-cuda",
        ModelFamily::Qwen3,
        "SYNAPSE_CUDA_PLAN_QWEN3_PACKAGE",
    );
}

fn check_family(profile_id: &str, family: ModelFamily, package_var: &str) {
    let package = std::env::var(package_var)
        .unwrap_or_else(|_| panic!("set {package_var} to the converted {profile_id} package path"));
    let with_graphs = drive(profile_id, family, &package, true);
    let without_graphs = drive(profile_id, family, &package, false);
    assert_eq!(with_graphs.len(), without_graphs.len());
    for ((length, on), (_, off)) in with_graphs.iter().zip(&without_graphs) {
        assert!(
            same_bits(on, off),
            "{profile_id} length {length}: graph replay on and off returned different vectors"
        );
    }
}

/// Loads the model through the production profile path, runs both cycles and
/// returns each length's vector in order. Asserts the memory bound and the
/// miss/hit identity along the way.
fn drive(
    profile_id: &str,
    family: ModelFamily,
    package: &str,
    graphs: bool,
) -> Vec<(usize, Vec<f32>)> {
    let profile = Profile::select(profile_id, Some("embed"))
        .unwrap_or_else(|error| panic!("select {profile_id}: {error}"));
    let mut engine = OwnedCudaEmbedEngine::serving(family);
    let mut config = RuntimeConfig::default();
    for (key, value) in [
        ("model_path", package),
        ("artifact_path", package),
        ("profile", profile_id),
        ("operation", "embed"),
        ("cuda_graphs", if graphs { "1" } else { "0" }),
    ] {
        config.values.insert(key.to_owned(), value.to_owned());
    }
    let artifact = ValidatedArtifact {
        digest: profile.package_digest.clone(),
        format: "safetensors-package".to_owned(),
    };
    let model = engine
        .load(&artifact, &config)
        .unwrap_or_else(|error| panic!("load {profile_id} graphs={graphs}: {error:?}"));

    let mut vectors = Vec::new();
    let after_first = run_cycle(&engine, &model, 0, graphs, &mut vectors);
    let after_second = run_cycle(&engine, &model, 1, graphs, &mut vectors);
    println!(
        "{profile_id} graphs={graphs}: device used after cycle 1 = {} MiB, after cycle 2 = {} MiB, slack = {} MiB",
        after_first >> 20,
        after_second >> 20,
        SLACK_BYTES >> 20
    );
    assert!(
        after_second <= after_first + SLACK_BYTES,
        "{profile_id} graphs={graphs}: device memory grew from {after_first} to {after_second} bytes \
         across five new shapes; the shape-plan cache is not bounded"
    );
    engine.unload(&model);
    drop(engine);
    vectors
}

/// Embeds each cycle length twice (a cache miss, then a hit) and returns the
/// device memory in use afterwards.
fn run_cycle(
    engine: &OwnedCudaEmbedEngine,
    model: &LoadedModel,
    extra_tokens: usize,
    graphs: bool,
    vectors: &mut Vec<(usize, Vec<f32>)>,
) -> u64 {
    for base in LENGTHS {
        let length = base + extra_tokens;
        let ids = token_ids(length);
        let miss = engine
            .embed_one(model, ids.clone())
            .unwrap_or_else(|error| panic!("length {length} graphs={graphs} miss: {error:?}"));
        let hit = engine
            .embed_one(model, ids)
            .unwrap_or_else(|error| panic!("length {length} graphs={graphs} hit: {error:?}"));
        assert!(
            same_bits(&miss, &hit),
            "length {length} graphs={graphs}: the shape-miss result differs from the cached-plan result"
        );
        vectors.push((length, miss));
    }
    device_used_bytes()
}

/// Deterministic token ids inside both models' vocabularies.
fn token_ids(length: usize) -> Vec<u32> {
    (0..length)
        .map(|index| 1000 + ((index * 7919) % 20_000) as u32)
        .collect()
}

fn same_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Device-wide used memory on the current CUDA device. The engine binds its
/// device on the calling thread before every forward, so this reads the
/// device the model runs on.
#[cfg(feature = "cuda")]
fn device_used_bytes() -> u64 {
    extern "C" {
        fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    }
    let (mut free, mut total) = (0_usize, 0_usize);
    // SAFETY: both pointers refer to live locals that the call only writes.
    let status = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    assert_eq!(status, 0, "cudaMemGetInfo failed with status {status}");
    (total - free) as u64
}

#[cfg(not(feature = "cuda"))]
fn device_used_bytes() -> u64 {
    panic!("shape_plan_hardware needs a build with --features cuda")
}
