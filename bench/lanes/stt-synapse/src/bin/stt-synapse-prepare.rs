//! Model conversion is an explicit owner action, outside benchmark timing.
use std::{fs::OpenOptions, io::Write, path::PathBuf};

use anyhow::{bail, ensure, Result};
use clap::Parser;
use synapse_parity::manifest::{AllMarker, Fp32Tensors};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    backend: String,
    #[arg(long)]
    output: PathBuf,
}

fn selection(backend: &str) -> Result<(synapse_parity::manifest::Manifest, &'static str, bool)> {
    use stt_synapse::*;
    let mut manifest = manifest()?;
    let (profile, pinned) = match backend {
        "ane" => (ANE_PROFILE, true),
        "cuda-f16" => (CUDA_PROFILE, true),
        "vulkan-f16" => (VULKAN_PROFILE, true),
        "cuda-f32" => {
            // Match the CUDA branch's scratch converter: retain the production
            // profile metadata but select homogeneous F32 floating tensors.
            manifest
                .profiles
                .get_mut(CUDA_PROFILE)
                .unwrap()
                .fp32_tensors = Fp32Tensors::All(AllMarker::All);
            (CUDA_PROFILE, false)
        }
        "vulkan-f32" => {
            manifest = vulkan_f32_manifest()?;
            (VULKAN_F32_PROFILE, false)
        }
        _ => bail!("backend must be ane, cuda-f16/f32 or vulkan-f16/f32"),
    };
    Ok((manifest, profile, pinned))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing to overwrite model package");
    let (manifest, id, pinned) = selection(&args.backend)?;
    let bytes = synapse_parity::convert::convert_profile_file(&manifest, id, &args.checkpoint)?;
    let digest = stt_common::input::sha256(&bytes);
    if pinned {
        ensure!(
            manifest.profiles[id]
                .converted_package_digest
                .as_ref()
                .is_some_and(|pin| pin.trim_start_matches("sha256:") == digest),
            "converted package differs from pin"
        );
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?
        .write_all(&bytes)?;
    println!(
        "{}: {} bytes, sha256:{digest}, measurement_only={}",
        args.output.display(),
        bytes.len(),
        !pinned
    );
    Ok(())
}

#[test]
fn stt_synapse_package_selection_keeps_sealed_profiles_unchanged() {
    use stt_synapse::*;
    let original = manifest().unwrap();
    for (backend, id) in [
        ("ane", ANE_PROFILE),
        ("cuda-f16", CUDA_PROFILE),
        ("vulkan-f16", VULKAN_PROFILE),
    ] {
        let (copy, selected, pinned) = selection(backend).unwrap();
        assert_eq!(selected, id);
        assert!(pinned);
        assert_eq!(copy.profiles, original.profiles);
    }
    let (cuda, id, pinned) = selection("cuda-f32").unwrap();
    assert!(!pinned);
    assert_eq!(id, CUDA_PROFILE);
    let mut expected = original.profiles[CUDA_PROFILE].clone();
    expected.fp32_tensors = Fp32Tensors::All(AllMarker::All);
    assert_eq!(cuda.profiles[CUDA_PROFILE], expected);
    let (vulkan, id, pinned) = selection("vulkan-f32").unwrap();
    assert!(!pinned);
    assert_eq!(id, VULKAN_F32_PROFILE);
    assert_eq!(
        vulkan.profiles[VULKAN_PROFILE],
        original.profiles[VULKAN_PROFILE]
    );
    assert!(vulkan.profiles[id].converted_package_digest.is_none());
    assert_eq!(
        vulkan.profiles[id].storage_dtype,
        synapse_parity::manifest::DType::F32
    );
    assert_eq!(manifest().unwrap().profiles, original.profiles);
    assert!(selection("cpu").is_err());
}
