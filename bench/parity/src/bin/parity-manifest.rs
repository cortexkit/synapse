//! `parity-manifest`: validate, generate and convert from `models.json`.
//!
//! - `validate`: schema-only checks of `models.json` and the committed
//!   certification inputs beside it; no checkpoint is read.
//! - `generate --hf-cache <dir> [--convert] [--check]`: check every model
//!   against its pinned checkpoint in a Hugging Face hub cache, recompute the
//!   Vulkan floors and digests (and, with `--convert`, every converted-package
//!   digest), then rewrite `models.json`, or with `--check` fail if it would
//!   change.
//! - `seal [--check]`: recompute only the Vulkan floors and digests; no
//!   checkpoint is read.
//! - `convert --profile <id> --hf-cache <dir> --out <file>`: write one
//!   converted package and print its digest.
//! - `reconvert --hf-cache <dir> [--profile <id>]`: reconvert worker profiles
//!   in memory and fail unless each digest equals the pinned one.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use synapse_parity::canonical::sha256_hex;
use synapse_parity::checkpoint::{check_checkpoint, snapshot_dir};
use synapse_parity::convert::convert_profile_file;
use synapse_parity::manifest::{Manifest, MANIFEST_FILE};
use synapse_parity::validate::{validate_dir, validate_manifest};
use synapse_parity::vulkan::{vulkan_floors, FLOOR_SEQUENCES};
use synapse_parity::{parity_dir, perr, Result};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("parity-manifest: {error}");
            ExitCode::FAILURE
        }
    }
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn option(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1).cloned())
}

fn required(args: &[String], name: &str) -> Result<String> {
    option(args, name).ok_or_else(|| perr!("missing {name} <value>"))
}

fn run(args: &[String]) -> Result<()> {
    let dir = option(args, "--dir")
        .map(PathBuf::from)
        .unwrap_or_else(parity_dir);
    match args.first().map(String::as_str) {
        Some("validate") => {
            let manifest = validate_dir(&dir)?;
            synapse_parity::validate::validate_inventory(&dir)?;
            println!("models.json valid; manifest digest {}", manifest.manifest_digest());
            Ok(())
        }
        Some("seal") => seal(&dir, None, false, flag(args, "--check")),
        Some("generate") => {
            let cache = PathBuf::from(required(args, "--hf-cache")?);
            seal(&dir, Some(&cache), flag(args, "--convert"), flag(args, "--check"))
        }
        Some("convert") => {
            let manifest = Manifest::load(&dir.join(MANIFEST_FILE))?;
            let profile = required(args, "--profile")?;
            let cache = PathBuf::from(required(args, "--hf-cache")?);
            let out = PathBuf::from(required(args, "--out")?);
            let bytes = convert_one(&manifest, &profile, &cache)?;
            std::fs::write(&out, &bytes).map_err(|error| perr!("write {}: {error}", out.display()))?;
            println!("{profile} sha256:{}", sha256_hex(&bytes));
            Ok(())
        }
        Some("reconvert") => {
            let manifest = Manifest::load(&dir.join(MANIFEST_FILE))?;
            let cache = PathBuf::from(required(args, "--hf-cache")?);
            let only = option(args, "--profile");
            let mut failures = Vec::new();
            for (id, profile) in &manifest.profiles {
                if !profile.lane.is_worker() || only.as_ref().is_some_and(|only| only != id) {
                    continue;
                }
                let digest = format!("sha256:{}", sha256_hex(&convert_one(&manifest, id, &cache)?));
                let pinned = profile.converted_package_digest.clone().unwrap_or_default();
                let verdict = if digest == pinned { "ok" } else { "MISMATCH" };
                println!("{verdict} {id} {digest}");
                if digest != pinned {
                    failures.push(id.clone());
                }
            }
            if failures.is_empty() {
                Ok(())
            } else {
                Err(perr!("reconversion differs from the pinned digest for {failures:?}"))
            }
        }
        _ => Err(perr!(
            "usage: parity-manifest validate | seal [--check] | generate --hf-cache <dir> [--convert] [--check] | convert --profile <id> --hf-cache <dir> --out <file> | reconvert --hf-cache <dir> [--profile <id>]"
        )),
    }
}

fn convert_one(manifest: &Manifest, profile_id: &str, cache: &Path) -> Result<Vec<u8>> {
    let profile = manifest
        .profiles
        .get(profile_id)
        .ok_or_else(|| perr!("unknown profile `{profile_id}`"))?;
    let model = manifest.model(&profile.model)?;
    let checkpoint = snapshot_dir(cache, model).join("model.safetensors");
    convert_profile_file(manifest, profile_id, &checkpoint)
}

fn seal(dir: &Path, cache: Option<&Path>, convert: bool, check: bool) -> Result<()> {
    let path = dir.join(MANIFEST_FILE);
    let original =
        std::fs::read(&path).map_err(|error| perr!("read {}: {error}", path.display()))?;
    let mut manifest = Manifest::from_slice(&original)?;
    if let Some(cache) = cache {
        for (slug, model) in &manifest.models {
            check_checkpoint(&manifest, slug, &snapshot_dir(cache, model), dir)?;
            eprintln!("checkpoint {slug}: ok");
        }
    }
    let context = u64::from(manifest.admission.max_context_tokens);
    let ids: Vec<String> = manifest.profiles.keys().cloned().collect();
    for id in &ids {
        let profile = manifest.profiles[id].clone();
        let model = manifest.model(&profile.model)?.clone();
        let mut updated = profile.clone();
        if let Some(sub_batch) = profile.vulkan_sub_batch_max_tokens {
            let floors = vulkan_floors(
                &model,
                profile.storage_dtype,
                &profile.fp32_tensors,
                context,
                FLOOR_SEQUENCES,
                u64::from(sub_batch),
            )?;
            updated.vulkan_min_storage_buffer_range = Some(floors.min_storage_buffer_range);
            updated.vulkan_min_device_local_bytes = Some(floors.min_device_local_bytes);
        }
        if convert && profile.lane.is_worker() {
            let cache = cache.ok_or_else(|| perr!("--convert needs --hf-cache"))?;
            let bytes = convert_one(&manifest, id, cache)?;
            let digest = format!("sha256:{}", sha256_hex(&bytes));
            eprintln!("converted {id}: {digest} ({} bytes)", bytes.len());
            updated.converted_package_digest = Some(digest);
        }
        manifest.profiles.insert(id.clone(), updated);
    }
    manifest.digests = manifest.computed_digests()?;
    validate_manifest(&manifest, dir)?;
    let rendered = manifest.to_pretty_bytes();
    if check {
        if rendered != original {
            return Err(perr!("{} is stale; rerun without --check", path.display()));
        }
        println!(
            "models.json is current; manifest digest {}",
            manifest.manifest_digest()
        );
    } else {
        std::fs::write(&path, &rendered)
            .map_err(|error| perr!("write {}: {error}", path.display()))?;
        println!(
            "wrote {}; manifest digest {}",
            path.display(),
            manifest.manifest_digest()
        );
    }
    Ok(())
}
