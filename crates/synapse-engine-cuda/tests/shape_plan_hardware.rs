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
//!
//! Optional variables for narrowing down a failure:
//!
//! - `SYNAPSE_CUDA_PLAN_LENGTHS=513,1025` runs exactly these lengths, in
//!   order, once each (miss then hit), instead of the two default cycles.
//!   The memory-growth assertion needs the two default cycles, so it is
//!   skipped; used memory is still printed.
//! - `SYNAPSE_CUDA_PLAN_GRAPHS=on|off|both` (default `both`) picks which graph
//!   replay modes run; the on/off comparison only happens with `both`.
//! - `SYNAPSE_CUDA_PLAN_POISON_MIB=4096` allocates that many MiB of device
//!   memory before each model load, fills it with 0xFF bytes and frees it.
//!   CUDA usually hands freed memory back to later allocations in the same
//!   process without clearing it, so the model's plans then start on 0xFF
//!   bytes instead of the zeroed pages a fresh process usually gets. A
//!   forward that reads memory nothing wrote then changes its result.

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
    let modes = graph_modes();
    let runs: Vec<_> = modes
        .iter()
        .map(|&graphs| drive(profile_id, family, &package, graphs))
        .collect();
    if let [with_graphs, without_graphs] = runs.as_slice() {
        assert_eq!(with_graphs.len(), without_graphs.len());
        for ((length, on), (_, off)) in with_graphs.iter().zip(without_graphs) {
            assert!(
                same_bits(on, off),
                "{profile_id} length {length}: graph replay on and off returned different vectors"
            );
        }
    }
}

/// Loads the model through the production profile path, runs the lengths and
/// returns each length's vector in order. Asserts the memory bound (default
/// cycles only) and the miss/hit identity along the way.
fn drive(
    profile_id: &str,
    family: ModelFamily,
    package: &str,
    graphs: bool,
) -> Vec<(usize, Vec<f32>)> {
    let (mut engine, model) = load_engine(profile_id, family, package, graphs);

    let mut vectors = Vec::new();
    if let Some(lengths) = lengths_override() {
        let used = run_lengths(&engine, &model, &lengths, graphs, &mut vectors);
        println!(
            "{profile_id} graphs={graphs}: lengths {lengths:?} passed; device used = {} MiB",
            used >> 20
        );
    } else {
        let first: Vec<usize> = LENGTHS.to_vec();
        let second: Vec<usize> = LENGTHS.iter().map(|length| length + 1).collect();
        let after_first = run_lengths(&engine, &model, &first, graphs, &mut vectors);
        let after_second = run_lengths(&engine, &model, &second, graphs, &mut vectors);
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
    }
    engine.unload(&model);
    drop(engine);
    vectors
}

/// Loads one model through the production profile path (profile selection,
/// header and digest validation, hardware floor), as the CUDA worker does.
fn load_engine(
    profile_id: &str,
    family: ModelFamily,
    package: &str,
    graphs: bool,
) -> (OwnedCudaEmbedEngine, LoadedModel) {
    let profile = Profile::select(profile_id, Some("embed"))
        .unwrap_or_else(|error| panic!("select {profile_id}: {error}"));
    if let Some(mib) = env_number("SYNAPSE_CUDA_PLAN_POISON_MIB") {
        poison_free_device_memory(mib);
    }
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
    (engine, model)
}

/// Lengths that reproduced a Qwen3 failure on an RTX 3060: the first five
/// build and evict plans for longer shapes, then 513 is a new, shorter shape
/// whose first forward and first graph replay disagreed. Each length on its
/// own in a new process passed.
const AFTER_EVICTIONS: [usize; 6] = [512, 1024, 2048, 3072, 4096, 513];

/// Embeds the last length of the list in a newly loaded engine first, as the
/// reference, then runs the whole list in a second engine and requires the
/// last length's result (its cache miss and its cache hit) to equal the
/// reference bit for bit. With graph replay on, a failure of the engine's own
/// capture check surfaces as a panic carrying the engine's diagnosis.
#[test]
#[ignore = "needs an NVIDIA GPU, --features cuda and SYNAPSE_CUDA_PLAN_QWEN3_PACKAGE"]
fn qwen3_shape_after_evictions_matches_a_fresh_engine() {
    let profile_id = "qwen3-embedding-0.6b.owned-cuda";
    let package_var = "SYNAPSE_CUDA_PLAN_QWEN3_PACKAGE";
    let package = std::env::var(package_var)
        .unwrap_or_else(|_| panic!("set {package_var} to the converted {profile_id} package path"));
    let lengths = lengths_override().unwrap_or_else(|| AFTER_EVICTIONS.to_vec());
    let last = *lengths.last().expect("at least one length");
    for graphs in graph_modes() {
        let (mut fresh, fresh_model) =
            load_engine(profile_id, ModelFamily::Qwen3, &package, graphs);
        let reference = fresh
            .embed_one(&fresh_model, token_ids(last))
            .unwrap_or_else(|error| panic!("fresh length {last} graphs={graphs}: {error:?}"));
        fresh.unload(&fresh_model);
        drop(fresh);

        let (mut engine, model) = load_engine(profile_id, ModelFamily::Qwen3, &package, graphs);
        let mut vectors = Vec::new();
        run_lengths(&engine, &model, &lengths, graphs, &mut vectors);
        engine.unload(&model);
        drop(engine);
        let (_, after) = vectors.last().expect("one vector per length");
        assert!(
            same_bits(after, &reference),
            "length {last} graphs={graphs}: the result after {lengths:?} differs from a fresh engine's result"
        );
        println!(
            "length {last} graphs={graphs}: result after {lengths:?} equals the fresh engine's"
        );
    }
}

fn graph_modes() -> Vec<bool> {
    match std::env::var("SYNAPSE_CUDA_PLAN_GRAPHS").as_deref() {
        Err(_) | Ok("both") => vec![true, false],
        Ok("on") => vec![true],
        Ok("off") => vec![false],
        Ok(other) => panic!("SYNAPSE_CUDA_PLAN_GRAPHS must be on, off or both, got {other}"),
    }
}

/// Embeds each length twice (a cache miss, then a hit) and returns the device
/// memory in use afterwards.
fn run_lengths(
    engine: &OwnedCudaEmbedEngine,
    model: &LoadedModel,
    lengths: &[usize],
    graphs: bool,
    vectors: &mut Vec<(usize, Vec<f32>)>,
) -> u64 {
    for &length in lengths {
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
        println!("length {length} graphs={graphs}: miss and hit bit-identical");
        vectors.push((length, miss));
    }
    device_used_bytes()
}

fn lengths_override() -> Option<Vec<usize>> {
    let raw = std::env::var("SYNAPSE_CUDA_PLAN_LENGTHS").ok()?;
    let lengths: Vec<usize> = raw
        .split(',')
        .map(|item| {
            item.trim().parse().unwrap_or_else(|_| {
                panic!("SYNAPSE_CUDA_PLAN_LENGTHS item {item:?} is not a length")
            })
        })
        .collect();
    assert!(
        lengths.iter().all(|&length| length > 0),
        "SYNAPSE_CUDA_PLAN_LENGTHS must list positive lengths"
    );
    Some(lengths)
}

fn env_number(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    Some(
        raw.trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a whole number, got {raw:?}")),
    )
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

/// Allocates `mib` MiB on the current device, fills it with 0xFF bytes and
/// frees it, so later allocations in this process likely reuse dirty memory.
/// Then reports how much of a new 64 MiB allocation still holds 0xFF.
#[cfg(feature = "cuda")]
fn poison_free_device_memory(mib: usize) {
    extern "C" {
        fn cudaMalloc(pointer: *mut *mut core::ffi::c_void, bytes: usize) -> i32;
        fn cudaMemset(pointer: *mut core::ffi::c_void, value: i32, bytes: usize) -> i32;
        fn cudaDeviceSynchronize() -> i32;
        fn cudaFree(pointer: *mut core::ffi::c_void) -> i32;
        fn cudaMemcpy(
            target: *mut core::ffi::c_void,
            source: *const core::ffi::c_void,
            bytes: usize,
            kind: i32,
        ) -> i32;
    }
    let bytes = mib << 20;
    let mut pointer = core::ptr::null_mut();
    // SAFETY: the out-pointer is a live local; the memset and free only touch
    // the block this call just allocated.
    unsafe {
        let status = cudaMalloc(&mut pointer, bytes);
        assert_eq!(
            status, 0,
            "cudaMalloc({mib} MiB) for poisoning failed with status {status}"
        );
        let status = cudaMemset(pointer, 0xFF, bytes);
        assert_eq!(
            status, 0,
            "cudaMemset for poisoning failed with status {status}"
        );
        let status = cudaDeviceSynchronize();
        assert_eq!(
            status, 0,
            "cudaDeviceSynchronize after poisoning failed with status {status}"
        );
        let status = cudaFree(pointer);
        assert_eq!(
            status, 0,
            "cudaFree after poisoning failed with status {status}"
        );
        // Check that a fresh allocation really comes back dirty on this
        // driver; if it does not, a passing poisoned run proves nothing.
        let probe_bytes = 64 << 20;
        let mut probe = core::ptr::null_mut();
        let status = cudaMalloc(&mut probe, probe_bytes);
        assert_eq!(
            status, 0,
            "cudaMalloc for the poison probe failed with status {status}"
        );
        let mut sample = vec![0_u8; 1 << 20];
        let status = cudaMemcpy(sample.as_mut_ptr().cast(), probe, sample.len(), 2);
        assert_eq!(
            status, 0,
            "cudaMemcpy for the poison probe failed with status {status}"
        );
        let status = cudaFree(probe);
        assert_eq!(
            status, 0,
            "cudaFree for the poison probe failed with status {status}"
        );
        let dirty = sample.iter().filter(|&&byte| byte == 0xFF).count();
        println!(
            "poison probe: {dirty} of {} sampled bytes of a new allocation are 0xFF",
            sample.len()
        );
    }
    println!("poisoned and freed {mib} MiB of device memory before load");
}

#[cfg(not(feature = "cuda"))]
fn device_used_bytes() -> u64 {
    panic!("shape_plan_hardware needs a build with --features cuda")
}

#[cfg(not(feature = "cuda"))]
fn poison_free_device_memory(_mib: usize) {
    panic!("shape_plan_hardware needs a build with --features cuda")
}
