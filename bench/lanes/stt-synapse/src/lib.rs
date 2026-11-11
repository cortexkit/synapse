//! Benchmark-only direct engines; no module worker IPC or certification capture.
use std::path::Path;

use anyhow::{bail, ensure, Result};
use synapse_speech::Features;

#[cfg(all(feature = "ane", target_os = "macos"))]
#[allow(dead_code)]
#[path = "../../../../crates/synapse-worker-ane-direct/src/backend.rs"]
mod backend;
#[cfg(all(feature = "ane", target_os = "macos"))]
#[allow(dead_code)]
#[path = "../../../../crates/synapse-worker-ane-direct/src/modernbert.rs"]
mod modernbert;
#[cfg(all(feature = "ane", target_os = "macos"))]
#[allow(dead_code)]
#[path = "../../../../crates/synapse-worker-ane-direct/src/parakeet.rs"]
mod parakeet;
#[cfg(all(feature = "ane", target_os = "macos"))]
#[allow(dead_code)]
#[path = "../../../../crates/synapse-worker-ane-direct/src/profile.rs"]
mod profile;
#[cfg(all(feature = "ane", target_os = "macos"))]
#[allow(dead_code)]
#[path = "../../../../crates/synapse-worker-ane-direct/src/qwen.rs"]
mod qwen;

pub const CUDA_PROFILE: &str = "parakeet-tdt-0.6b-v3.owned-cuda";
pub const VULKAN_PROFILE: &str = "parakeet-tdt-0.6b-v3.owned-vulkan";
pub const VULKAN_F32_PROFILE: &str = "parakeet-tdt-0.6b-v3.owned-vulkan.measurement-f32";
pub const ANE_PROFILE: &str = "parakeet-tdt-0.6b-v3.ane-direct-worker";

pub trait Engine {
    fn tokens(&mut self, features: Features) -> Result<Vec<u32>>;
}

pub fn load(name: &str, path: &Path) -> Result<Box<dyn Engine>> {
    match name {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "synapse-metal" => metal::load(path),
        #[cfg(all(feature = "ane", target_os = "macos"))]
        "synapse-ane" => direct_ane::load(path),
        #[cfg(feature = "cuda")]
        "synapse-cuda-f16" | "synapse-cuda-f32" => cuda::load(path, name.ends_with("f32")),
        #[cfg(feature = "vulkan")]
        "synapse-vulkan-f16" | "synapse-vulkan-f32" => vulkan::load(path, name.ends_with("f32")),
        _ => {
            let _ = path;
            bail!("backend {name} is unavailable in this build/platform; enable its Cargo feature")
        }
    }
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
fn request(features: Features) -> synapse_core::TranscribeRequest {
    synapse_core::TranscribeRequest {
        mel: features.values,
        mel_frames: features.mel_frames,
        valid_mel_frames: features.valid_mel_frames,
        observe: false,
    }
}

pub fn token_ids(output: synapse_core::TranscribeOutput) -> Result<Vec<u32>> {
    ensure!(
        output.observation.is_none(),
        "benchmark engine returned observations"
    );
    output
        .triples
        .iter()
        .map(|t| Ok(u32::try_from(t.token_id)?))
        .collect()
}

pub fn package(path: &Path) -> std::path::PathBuf {
    if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.into()
    }
}

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(feature = "ane", target_os = "macos"))]
mod direct_ane;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
#[cfg(feature = "vulkan")]
mod vulkan;

pub fn manifest() -> Result<synapse_parity::manifest::Manifest> {
    Ok(serde_json::from_slice(include_bytes!(
        "../../../parity/models.json"
    ))?)
}

/// The Vulkan branch's measurement profile is deliberately not a sealed
/// production profile. Reproduce its local clone only for bench conversion/load.
pub fn vulkan_f32_manifest() -> Result<synapse_parity::manifest::Manifest> {
    use synapse_parity::manifest::{AllMarker, DType, Fp32Tensors};
    let mut manifest = manifest()?;
    let mut profile = manifest.profiles[VULKAN_PROFILE].clone();
    profile.fp32_tensors = Fp32Tensors::All(AllMarker::All);
    profile.storage_dtype = DType::F32;
    profile.compute_dtype = DType::F32;
    profile.converted_package_digest = None;
    #[cfg(feature = "vulkan")]
    {
        use synapse_worker_vulkan::transcribe::{floors_for_precision, maximum, WeightPrecision};
        let required = floors_for_precision(
            &manifest.models[&profile.model],
            maximum(&profile)?,
            WeightPrecision::F32,
        )?;
        profile.vulkan_min_device_local_bytes = Some(required.min_device_local_bytes);
        profile.vulkan_min_storage_buffer_range = Some(required.min_storage_buffer_range);
    }
    manifest.profiles.insert(VULKAN_F32_PROFILE.into(), profile);
    Ok(manifest)
}
