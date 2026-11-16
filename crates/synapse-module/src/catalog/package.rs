//! Named source-package assembly for the Qwen3-ASR catalog lane.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{CatalogBackend, CatalogEntry, CatalogFile};
use crate::{artifact_invalid_error, catalog_artifact_error, io_to_load_error, sha256_file};
use crate::{ModelCache, WireOperationError};

/// Assemble only the ten original Qwen3-ASR source files pinned in the parity
/// manifest, retaining their names (including the shard index). This is not a
/// serving artifact: later ASR loading must use `load_verified_asr_artifacts`
/// from `owned_decode_routing` for the derived 8-bit quantized decoder and
/// half-precision encoder/embedding files. The legacy
/// owned-model assembler remains single-file for every other family/format.
pub(crate) fn assemble_source_package(
    entry: &CatalogEntry,
    backend: &CatalogBackend,
    cache: &ModelCache,
) -> Result<PathBuf, WireOperationError> {
    validate_source_pins(entry, backend)?;
    assemble_verified_files(entry, backend, cache)
}

fn validate_source_pins(
    entry: &CatalogEntry,
    backend: &CatalogBackend,
) -> Result<(), WireOperationError> {
    if backend.family.as_deref() != Some("qwen3-asr")
        || entry.backend(&backend.backend) != Some(backend)
        || entry.id != "qwen3-asr-1.7b"
    {
        return Err(artifact_invalid_error(
            "multi-file source packages require the Qwen3-ASR catalog family",
        ));
    }
    let manifest = synapse_parity::manifest::Manifest::from_slice(include_bytes!(
        "../../../../bench/parity/models.json"
    ))
    .map_err(|error| artifact_invalid_error(error.to_string()))?;
    let model = manifest
        .model(&entry.id)
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    let files = entry
        .backend_named_files(&backend.backend)
        .collect::<Vec<_>>();
    if entry.upstream.hf_repo != model.hf_repo
        || entry.upstream.revision != model.hf_revision
        || files.len() != model.files.len()
        || files
            .iter()
            .map(|(name, _)| *name)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != files.len()
        || files
            .iter()
            .any(|(name, file)| model.files.get(*name) != Some(&file.sha256))
    {
        return Err(artifact_invalid_error(
            "Qwen3-ASR source package differs from the pinned source manifest",
        ));
    }
    super::validate_file_roles(entry).map_err(|error| artifact_invalid_error(error.to_string()))
}

fn verify_file(path: &Path, file: &CatalogFile) -> Result<(), WireOperationError> {
    let size = fs::metadata(path).ok().map(|metadata| metadata.len());
    let digest = sha256_file(path).ok();
    if size != Some(file.size_bytes) || digest.as_deref() != Some(file.sha256.as_str()) {
        return Err(catalog_artifact_error(file, digest, size));
    }
    Ok(())
}

fn assemble_verified_files(
    entry: &CatalogEntry,
    backend: &CatalogBackend,
    cache: &ModelCache,
) -> Result<PathBuf, WireOperationError> {
    let files = entry
        .backend_named_files(&backend.backend)
        .collect::<BTreeMap<_, _>>();
    // Hash source blobs even when an assembled directory exists: a previously
    // trusted cache record must not hide a missing or modified shard.
    for file in files.values() {
        verify_file(&cache.blob_path(&file.sha256), file)?;
    }
    let packages = cache.root().join("owned-metal-models");
    let key = entry.manifest_digest();
    let package = packages.join(&key);
    if package.is_dir() {
        for (name, file) in &files {
            verify_file(&package.join(name), file)?;
        }
        return Ok(package);
    }
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let temporary = packages.join(format!(
        ".{key}.{}.{}.tmp",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&temporary)
        .map_err(|error| io_to_load_error("create ASR source package", &temporary, &error))?;
    let result = (|| {
        for (name, file) in &files {
            let destination = temporary.join(name);
            fs::copy(cache.blob_path(&file.sha256), &destination)
                .map_err(|error| io_to_load_error("copy ASR source file", &destination, &error))?;
            verify_file(&destination, file)?;
        }
        // Reuse the existing package garbage collector's roots descriptor, so
        // every shard and sidecar keeps the assembled directory alive.
        let mut digests = files.values().map(|file| &file.sha256).collect::<Vec<_>>();
        digests.sort();
        digests.dedup();
        let roots = temporary.join("role-digests.json");
        fs::write(
            &roots,
            serde_json::to_vec(&digests).expect("source digests"),
        )
        .map_err(|error| io_to_load_error("write ASR source roots", &roots, &error))?;
        match fs::rename(&temporary, &package) {
            Ok(()) => {}
            Err(_) if package.is_dir() => {
                for (name, file) in &files {
                    verify_file(&package.join(name), file)?;
                }
            }
            Err(error) => {
                return Err(io_to_load_error(
                    "publish ASR source package",
                    &package,
                    &error,
                ))
            }
        }
        Ok(package)
    })();
    let _ = fs::remove_dir_all(&temporary);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256_hex;

    fn fixture() -> (CatalogEntry, CatalogBackend) {
        let mut entry = super::super::compiled_catalog().unwrap().models[0].clone();
        let mut backend = entry.backends.remove(0);
        backend.family = Some("qwen3-asr".into());
        entry.id = "qwen3-asr-1.7b".into();
        entry.backends = vec![backend.clone()];
        let manifest = synapse_parity::manifest::Manifest::from_slice(include_bytes!(
            "../../../../bench/parity/models.json"
        ))
        .unwrap();
        let model = manifest.model(&entry.id).unwrap();
        entry.upstream.hf_repo = model.hf_repo.clone();
        entry.upstream.revision = model.hf_revision.clone();
        entry.files = model
            .files
            .iter()
            .map(|(name, digest)| CatalogFile {
                path: name.clone(),
                role: if name.ends_with(".safetensors") {
                    "model"
                } else if name == "vocab.json" {
                    "tokenizer"
                } else {
                    "config"
                }
                .into(),
                sha256: digest.clone(),
                size_bytes: 0,
                backends: vec![backend.backend.clone()],
            })
            .collect();
        (entry, backend)
    }

    struct Installed {
        root: PathBuf,
        cache: ModelCache,
        entry: CatalogEntry,
        backend: CatalogBackend,
    }

    impl Installed {
        fn new() -> Self {
            static SEQUENCE: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "synapse-catalog-multishard-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            let cache = ModelCache::new(&root);
            fs::create_dir_all(root.join("blobs")).unwrap();
            let (mut entry, backend) = fixture();
            for file in &mut entry.files {
                let body = format!("pinned fixture for {}", file.path);
                file.sha256 = sha256_hex(body.as_bytes());
                file.size_bytes = body.len() as u64;
                fs::write(cache.blob_path(&file.sha256), body).unwrap();
            }
            Self {
                root,
                cache,
                entry,
                backend,
            }
        }

        fn assemble(&self) -> Result<PathBuf, WireOperationError> {
            assemble_verified_files(&self.entry, &self.backend, &self.cache)
        }

        fn shard(&self) -> &CatalogFile {
            self.entry
                .files
                .iter()
                .find(|file| file.path == "model-00001-of-00002.safetensors")
                .unwrap()
        }
    }

    impl Drop for Installed {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }

    #[test]
    fn catalog_multishard_named_lookup_and_role_cardinality() {
        let (mut entry, _) = fixture();
        super::super::validate_file_roles(&entry).unwrap();
        assert_eq!(entry.backend_named_files("metal").count(), 10);
        assert_eq!(entry.backend_named_files("cuda").count(), 0);
        for family in ["qwen3", "gte-modernbert", "parakeet-tdt", "unknown"] {
            entry.backends[0].family = Some(family.into());
            assert!(matches!(
                super::super::validate_file_roles(&entry),
                Err(super::super::CatalogError::RoleCardinality {
                    role: "model",
                    count: 2,
                    ..
                })
            ));
            entry
                .files
                .iter_mut()
                .filter(|file| file.role == "model")
                .skip(1)
                .for_each(|file| file.role = "config".into());
            assert!(matches!(
                super::super::validate_file_roles(&entry),
                Err(super::super::CatalogError::RoleCardinality { role: "config", .. })
            ));
            entry = fixture().0;
        }
        entry.files.retain(|file| file.role != "model");
        assert!(matches!(
            super::super::validate_file_roles(&entry),
            Err(super::super::CatalogError::RoleCardinality {
                role: "model",
                count: 0,
                ..
            })
        ));
        let (mut entry, _) = fixture();
        entry.files[0].role = "tokenizer".into();
        assert!(matches!(
            super::super::validate_file_roles(&entry),
            Err(super::super::CatalogError::RoleCardinality {
                role: "tokenizer",
                count: 2,
                ..
            })
        ));
    }

    #[test]
    fn catalog_multishard_requires_exact_source_pins() {
        let (entry, backend) = fixture();
        validate_source_pins(&entry, &backend).unwrap();
        for change in 0..5 {
            let mut altered = entry.clone();
            match change {
                0 => {
                    altered.files.remove(0);
                }
                1 => altered.files[0].sha256 = "0".repeat(64),
                2 => altered.files[0].path = "../unlisted.json".into(),
                3 => altered.upstream.revision = "0".repeat(40),
                _ => altered.files.push(altered.files[0].clone()),
            }
            assert_eq!(
                validate_source_pins(&altered, &backend).unwrap_err().code,
                "artifact_invalid"
            );
        }
        for family in ["qwen3", "parakeet-tdt", "gte-modernbert"] {
            let mut altered = entry.clone();
            altered.backends[0].family = Some(family.into());
            let cache = ModelCache::new("unused-cache");
            assert_eq!(
                assemble_source_package(&altered, &altered.backends[0], &cache)
                    .unwrap_err()
                    .code,
                "artifact_invalid"
            );
        }
    }

    #[test]
    fn catalog_multishard_legacy_owned_assembler_still_refuses_extra_files() {
        let (root, state) =
            crate::catalog_runtime_tests::isolated_catalog_state("multishard-legacy", false);
        let entry = state
            .runtime
            .release_catalog
            .entry("gte-modernbert-base")
            .unwrap();
        let backend = entry.backend("metal").unwrap();
        let spec = crate::catalog_native_lane_spec(
            &state,
            entry,
            backend,
            format!("sha256:{}", "0".repeat(64)),
        )
        .unwrap();
        let mut profile = crate::stored_owned_profile(&spec).unwrap().unwrap();
        profile.extra_locators.push(spec.model_locator.clone());
        let model_path = state
            .model_cache
            .blob_path(&entry.backend_files("metal")["model"].sha256);
        let error =
            crate::assemble_owned_model_package(&spec, &model_path, &state.model_cache, &profile)
                .unwrap_err();
        assert_eq!(error.code, "artifact_invalid");
        assert!(error.message.contains("sharded"));
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_multishard_assembles_every_named_file_and_roots_each_digest() {
        let installed = Installed::new();
        let package = installed.assemble().unwrap();
        assert_eq!(fs::read_dir(&package).unwrap().count(), 11);
        for file in &installed.entry.files {
            assert_eq!(sha256_file(&package.join(&file.path)).unwrap(), file.sha256);
        }
        let roots: Vec<String> =
            serde_json::from_slice(&fs::read(package.join("role-digests.json")).unwrap()).unwrap();
        assert_eq!(roots.len(), 10);
        assert!(installed
            .entry
            .files
            .iter()
            .all(|file| roots.contains(&file.sha256)));
        assert_eq!(installed.assemble().unwrap(), package);
    }

    #[test]
    fn catalog_multishard_missing_shard_is_typed_even_after_assembly() {
        for cached in [false, true] {
            let installed = Installed::new();
            if cached {
                installed.assemble().unwrap();
            }
            fs::remove_file(installed.cache.blob_path(&installed.shard().sha256)).unwrap();
            let error = installed.assemble().unwrap_err();
            assert_eq!(error.code, "artifact_invalid");
            let details = error.details.unwrap();
            assert_eq!(details["file"], installed.shard().path);
            assert!(details["actual_sha256"].is_null());
            assert_eq!(details["recovery_op"], "models.download");
        }
    }

    #[test]
    fn catalog_multishard_corrupt_shard_is_typed_even_at_same_size() {
        let installed = Installed::new();
        installed.assemble().unwrap();
        fs::write(
            installed.cache.blob_path(&installed.shard().sha256),
            vec![b'x'; installed.shard().size_bytes as usize],
        )
        .unwrap();
        let error = installed.assemble().unwrap_err();
        assert_eq!(error.code, "artifact_invalid");
        let details = error.details.unwrap();
        assert_eq!(details["file"], installed.shard().path);
        assert_eq!(details["actual_size"], installed.shard().size_bytes);
        assert_ne!(details["actual_sha256"], details["expected_sha256"]);
    }

    #[test]
    fn catalog_multishard_cached_package_is_rehashed() {
        let installed = Installed::new();
        let package = installed.assemble().unwrap();
        fs::write(package.join(&installed.shard().path), b"modified").unwrap();
        let error = installed.assemble().unwrap_err();
        assert_eq!(error.code, "artifact_invalid");
        assert_eq!(error.details.unwrap()["file"], installed.shard().path);
    }
}
