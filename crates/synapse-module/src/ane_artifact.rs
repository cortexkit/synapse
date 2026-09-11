//! Stable, content-addressed materialization for archived Core ML bundles.

use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CACHE_DIRECTORY: &str = "ane-coreml";
const METADATA_FILE: &str = "materialization.json";
const METADATA_FORMAT: &str = "synapse-ane-coreml-materialization-v1";
const ABANDONED_TEMP_MIN_AGE: Duration = Duration::from_secs(24 * 60 * 60);
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MaterializedCoreMlArtifact {
    pub(crate) path: PathBuf,
    pub(crate) digest: String,
    pub(crate) reused: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MaterializationMetadata {
    format: String,
    source_digest: String,
    model_relative_path: String,
    materialized_digest: String,
}

/// Return a directory path suitable for Core ML, extracting archives only once.
///
/// The catalog digest remains the artifact's identity. The second digest returned
/// here covers the extracted directory and is only used by the worker to verify
/// the materialized transport bytes.
pub(crate) fn materialize_core_ml_artifact(
    source_path: &Path,
    source_digest: &str,
    cache_root: &Path,
) -> Result<MaterializedCoreMlArtifact> {
    materialize_with_extractor(source_path, source_digest, cache_root, extract_archive)
}

fn materialize_with_extractor<F>(
    source_path: &Path,
    source_digest: &str,
    cache_root: &Path,
    extractor: F,
) -> Result<MaterializedCoreMlArtifact>
where
    F: Fn(&Path, &Path) -> Result<()>,
{
    let source_digest = normalize_digest(source_digest)?;
    let source_metadata = fs::metadata(source_path)
        .with_context(|| format!("stat Core ML artifact {}", source_path.display()))?;
    if source_metadata.is_dir() {
        return Ok(MaterializedCoreMlArtifact {
            path: source_path.to_path_buf(),
            digest: source_digest,
            reused: true,
        });
    }
    ensure!(
        source_metadata.is_file(),
        "Core ML artifact is neither a file nor a directory: {}",
        source_path.display()
    );

    let actual_source_digest = format!("sha256:{}", sha256_file(source_path)?);
    ensure!(
        actual_source_digest == source_digest,
        "Core ML artifact digest mismatch for {}: expected {}, got {}",
        source_path.display(),
        source_digest,
        actual_source_digest
    );

    let cache_directory = cache_root.join(CACHE_DIRECTORY);
    fs::create_dir_all(&cache_directory).with_context(|| {
        format!(
            "create Core ML materialization cache {}",
            cache_directory.display()
        )
    })?;
    let final_root = entry_path_for_normalized_digest(&cache_directory, &source_digest);
    if let Some(existing) = load_existing(&final_root, &source_digest)? {
        return Ok(existing);
    }

    let source_key = digest_key(&source_digest);
    let mut transaction = TempTree::create(&cache_directory, source_key)?;
    let extraction_root = transaction.path().join("artifact");
    fs::create_dir(&extraction_root).with_context(|| {
        format!(
            "create private Core ML extraction directory {}",
            extraction_root.display()
        )
    })?;
    extractor(source_path, &extraction_root)?;

    let model_path = find_compiled_model(&extraction_root)?;
    let materialized_digest = format!("sha256:{}", sha256_directory(&model_path)?);
    let model_relative_path = model_path
        .strip_prefix(transaction.path())
        .expect("compiled model was found below the transaction root")
        .to_str()
        .context("Core ML materialized path is not valid UTF-8")?
        .to_string();
    let metadata = MaterializationMetadata {
        format: METADATA_FORMAT.to_string(),
        source_digest: source_digest.clone(),
        model_relative_path,
        materialized_digest: materialized_digest.clone(),
    };
    let metadata_path = transaction.path().join(METADATA_FILE);
    let mut metadata_file = File::create(&metadata_path).with_context(|| {
        format!(
            "create Core ML materialization metadata {}",
            metadata_path.display()
        )
    })?;
    metadata_file
        .write_all(
            &serde_json::to_vec_pretty(&metadata)
                .context("serialize Core ML materialization metadata")?,
        )
        .with_context(|| {
            format!(
                "write Core ML materialization metadata {}",
                metadata_path.display()
            )
        })?;
    metadata_file.sync_all().with_context(|| {
        format!(
            "sync Core ML materialization metadata {}",
            metadata_path.display()
        )
    })?;

    match fs::rename(transaction.path(), &final_root) {
        Ok(()) => {
            transaction.disarm();
            sync_directory(&cache_directory)?;
            Ok(MaterializedCoreMlArtifact {
                path: final_root.join(metadata.model_relative_path),
                digest: materialized_digest,
                reused: false,
            })
        }
        Err(rename_error) if final_root.is_dir() => {
            // Another process won publication. Its immutable digest-keyed tree is
            // authoritative; dropping the transaction removes only this loser's
            // private sibling.
            load_existing(&final_root, &source_digest)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "publish Core ML materialization {}: {}; concurrent result was incomplete",
                    final_root.display(),
                    rename_error
                )
            })
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "publish Core ML materialization {} -> {}",
                transaction.path().display(),
                final_root.display()
            )
        }),
    }
}

fn extract_archive(source_path: &Path, destination: &Path) -> Result<()> {
    let output = Command::new("/usr/bin/unzip")
        .arg("-q")
        .arg("-o")
        .arg(source_path)
        .arg("-d")
        .arg(destination)
        .output()
        .with_context(|| format!("launch unzip for {}", source_path.display()))?;
    ensure!(
        output.status.success(),
        "unzip Core ML artifact {} failed: {}",
        source_path.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

fn load_existing(
    final_root: &Path,
    expected_source_digest: &str,
) -> Result<Option<MaterializedCoreMlArtifact>> {
    if !final_root.exists() {
        return Ok(None);
    }
    ensure!(
        final_root.is_dir(),
        "Core ML materialization path is not a directory: {}",
        final_root.display()
    );
    let metadata_path = final_root.join(METADATA_FILE);
    let metadata: MaterializationMetadata =
        serde_json::from_slice(&fs::read(&metadata_path).with_context(|| {
            format!(
                "read Core ML materialization metadata {}",
                metadata_path.display()
            )
        })?)
        .with_context(|| {
            format!(
                "parse Core ML materialization metadata {}",
                metadata_path.display()
            )
        })?;
    ensure!(
        metadata.format == METADATA_FORMAT,
        "Core ML materialization {} has unsupported metadata format {}",
        final_root.display(),
        metadata.format
    );
    ensure!(
        metadata.source_digest == expected_source_digest,
        "Core ML materialization {} belongs to {}, expected {}",
        final_root.display(),
        metadata.source_digest,
        expected_source_digest
    );
    let materialized_digest = normalize_digest(&metadata.materialized_digest)?;
    let relative = validated_relative_path(&metadata.model_relative_path)?;
    let model_path = final_root.join(relative);
    ensure!(
        is_compiled_model(&model_path),
        "Core ML materialization {} does not contain a complete compiled model at {}",
        final_root.display(),
        model_path.display()
    );
    Ok(Some(MaterializedCoreMlArtifact {
        path: model_path,
        digest: materialized_digest,
        reused: true,
    }))
}

fn find_compiled_model(extraction_root: &Path) -> Result<PathBuf> {
    let mut directories = Vec::new();
    collect_directories(extraction_root, &mut directories)?;
    directories.sort_by(|left, right| left.as_os_str().cmp(right.as_os_str()));
    directories
        .into_iter()
        .find(|path| is_compiled_model(path))
        .with_context(|| {
            format!(
                "extracted Core ML artifact {} did not contain a compiled .mlmodelc bundle",
                extraction_root.display()
            )
        })
}

fn collect_directories(path: &Path, directories: &mut Vec<PathBuf>) -> Result<()> {
    directories.push(path.to_path_buf());
    for entry in fs::read_dir(path)
        .with_context(|| format!("list extracted Core ML directory {}", path.display()))?
    {
        let entry = entry
            .with_context(|| format!("read extracted Core ML entry below {}", path.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("stat extracted Core ML entry {}", entry.path().display()))?;
        ensure!(
            !file_type.is_symlink(),
            "extracted Core ML artifact contains unsupported symlink {}",
            entry.path().display()
        );
        if file_type.is_dir() {
            collect_directories(&entry.path(), directories)?;
        }
    }
    Ok(())
}

fn is_compiled_model(path: &Path) -> bool {
    path.is_dir() && (path.join("model.mil").exists() || path.join("model.mlmodel").exists())
}

fn sha256_directory(root: &Path) -> Result<String> {
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = Sha256::new();
    for (relative, path) in files {
        digest.update(relative.as_bytes());
        digest.update([0]);
        update_digest_from_file(&mut digest, &path)?;
    }
    Ok(hex::encode(digest.finalize()))
}

fn collect_files(root: &Path, path: &Path, files: &mut Vec<(String, PathBuf)>) -> Result<()> {
    for entry in fs::read_dir(path)
        .with_context(|| format!("list materialized Core ML directory {}", path.display()))?
    {
        let entry = entry
            .with_context(|| format!("read materialized Core ML entry below {}", path.display()))?;
        let entry_path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("stat materialized Core ML entry {}", entry_path.display()))?;
        ensure!(
            !file_type.is_symlink(),
            "materialized Core ML artifact contains unsupported symlink {}",
            entry_path.display()
        );
        if file_type.is_dir() {
            collect_files(root, &entry_path, files)?;
        } else if file_type.is_file() {
            let relative = entry_path
                .strip_prefix(root)
                .expect("collected file is below the materialized root")
                .to_str()
                .context("Core ML bundle contains a non-UTF-8 path")?
                .to_string();
            files.push((relative, entry_path));
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    update_digest_from_file(&mut digest, path)?;
    Ok(hex::encode(digest.finalize()))
}

fn update_digest_from_file(digest: &mut Sha256, path: &Path) -> Result<()> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {} for hashing", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(())
}

fn normalize_digest(value: &str) -> Result<String> {
    let raw = value.strip_prefix("sha256:").unwrap_or(value).trim();
    ensure!(
        raw.len() == 64 && raw.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "sha256 digest must be 64 hex characters, got '{value}'"
    );
    Ok(format!("sha256:{}", raw.to_ascii_lowercase()))
}

fn digest_key(normalized_digest: &str) -> &str {
    normalized_digest
        .strip_prefix("sha256:")
        .expect("normalized digest has a sha256 prefix")
}

fn entry_path_for_normalized_digest(cache_directory: &Path, normalized_digest: &str) -> PathBuf {
    cache_directory.join(digest_key(normalized_digest))
}

#[cfg(test)]
pub(crate) fn materialized_entry_path(cache_root: &Path, digest: &str) -> Result<PathBuf> {
    let digest = normalize_digest(digest)?;
    Ok(entry_path_for_normalized_digest(
        &cache_root.join(CACHE_DIRECTORY),
        &digest,
    ))
}

fn validated_relative_path(value: &str) -> Result<PathBuf> {
    let path = Path::new(value);
    ensure!(
        !path.as_os_str().is_empty(),
        "materialized model path is empty"
    );
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "materialized model path must stay below its digest root: {value}"
    );
    Ok(path.to_path_buf())
}

/// Remove a completed derivative only after model-cache GC deleted its source.
/// Private extraction siblings are also safe to remove then because source-cache
/// leases prevent publication and source deletion from running concurrently.
pub(crate) fn remove_for_source_digest(cache_root: &Path, digest: &str) -> Result<bool> {
    let digest = normalize_digest(digest)?;
    let cache_directory = cache_root.join(CACHE_DIRECTORY);
    let final_root = entry_path_for_normalized_digest(&cache_directory, &digest);
    let mut removed = false;
    if final_root.exists() {
        fs::remove_dir_all(&final_root).with_context(|| {
            format!(
                "remove Core ML materialization after source GC {}",
                final_root.display()
            )
        })?;
        removed = true;
    }
    if cache_directory.is_dir() {
        let prefix = format!(".{}.", digest_key(&digest));
        for entry in fs::read_dir(&cache_directory).with_context(|| {
            format!(
                "list Core ML materialization cache {}",
                cache_directory.display()
            )
        })? {
            let entry = entry.with_context(|| {
                format!(
                    "read Core ML materialization cache entry below {}",
                    cache_directory.display()
                )
            })?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && name.ends_with(".tmp") {
                fs::remove_dir_all(entry.path()).with_context(|| {
                    format!(
                        "remove private Core ML extraction after source GC {}",
                        entry.path().display()
                    )
                })?;
                removed = true;
            }
        }
        if removed {
            sync_directory(&cache_directory)?;
        }
    }
    Ok(removed)
}

/// Delete only abandoned private extraction trees. Completed digest roots are
/// retained until the model-cache GC deletes the matching source artifact.
pub(crate) fn cleanup_abandoned_temps(cache_root: &Path, now: SystemTime) -> Result<()> {
    let cache_directory = cache_root.join(CACHE_DIRECTORY);
    if !cache_directory.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(&cache_directory).with_context(|| {
        format!(
            "list Core ML materialization cache {}",
            cache_directory.display()
        )
    })? {
        let entry = entry.with_context(|| {
            format!(
                "read Core ML materialization cache entry below {}",
                cache_directory.display()
            )
        })?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with('.') || !name.ends_with(".tmp") {
            continue;
        }
        let metadata = entry.metadata().with_context(|| {
            format!("stat private Core ML extraction {}", entry.path().display())
        })?;
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let Ok(age) = now.duration_since(modified) else {
            continue;
        };
        if age >= ABANDONED_TEMP_MIN_AGE {
            fs::remove_dir_all(entry.path()).with_context(|| {
                format!(
                    "remove abandoned private Core ML extraction {}",
                    entry.path().display()
                )
            })?;
        }
    }
    Ok(())
}

pub(crate) fn total_materialized_bytes(cache_root: &Path) -> Result<u64> {
    let cache_directory = cache_root.join(CACHE_DIRECTORY);
    if !cache_directory.is_dir() {
        return Ok(0);
    }
    let mut total = 0_u64;
    for entry in fs::read_dir(&cache_directory).with_context(|| {
        format!(
            "list Core ML materialization cache {}",
            cache_directory.display()
        )
    })? {
        let entry = entry.with_context(|| {
            format!(
                "read Core ML materialization cache entry below {}",
                cache_directory.display()
            )
        })?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.len() != 64 || !name.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        total = total.saturating_add(directory_bytes(&entry.path())?);
    }
    Ok(total)
}

fn directory_bytes(path: &Path) -> Result<u64> {
    let mut total = 0_u64;
    for entry in fs::read_dir(path)
        .with_context(|| format!("list Core ML cache directory {}", path.display()))?
    {
        let entry =
            entry.with_context(|| format!("read Core ML cache entry below {}", path.display()))?;
        let metadata = entry
            .metadata()
            .with_context(|| format!("stat Core ML cache entry {}", entry.path().display()))?;
        if metadata.is_dir() {
            total = total.saturating_add(directory_bytes(&entry.path())?);
        } else if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("open directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("sync directory {}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

struct TempTree {
    path: PathBuf,
    armed: bool,
}

impl TempTree {
    fn create(cache_directory: &Path, source_key: &str) -> Result<Self> {
        for _ in 0..100 {
            let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = cache_directory.join(format!(
                ".{source_key}.{}.{}.{nonce}.tmp",
                std::process::id(),
                nanos
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path, armed: true }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("create private Core ML extraction {}", path.display())
                    })
                }
            }
        }
        bail!("could not allocate a unique private Core ML extraction directory")
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempTree {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier, Mutex};

    use super::*;

    #[test]
    fn digest_key_selects_distinct_paths_and_reuses_only_an_exact_digest() {
        let root = temp_root("digest-key");
        let cache = root.join("cache");
        let first_source = root.join("first.zip");
        let second_source = root.join("second.zip");
        fs::create_dir_all(&root).unwrap();
        fs::write(&first_source, b"first archive bytes").unwrap();
        fs::write(&second_source, b"second archive bytes").unwrap();
        let first_digest = file_digest(&first_source);
        let second_digest = file_digest(&second_source);
        let extracts = AtomicU64::new(0);
        let extractor = |source: &Path, destination: &Path| {
            extracts.fetch_add(1, Ordering::Relaxed);
            write_fake_model(destination, &fs::read(source)?)
        };

        let first = materialize_with_extractor(&first_source, &first_digest, &cache, extractor)
            .expect("first digest materializes");
        let first_reused =
            materialize_with_extractor(&first_source, &first_digest, &cache, extractor)
                .expect("same digest reuses the materialization");
        let second = materialize_with_extractor(&second_source, &second_digest, &cache, extractor)
            .expect("different digest materializes separately");

        assert_eq!(first.path, first_reused.path);
        assert!(first_reused.reused);
        assert_ne!(first.path, second.path);
        assert_ne!(first.digest, second.digest);
        assert!(
            first
                .path
                .to_string_lossy()
                .contains(digest_key(&first_digest)),
            "first path must carry its source digest"
        );
        assert!(
            second
                .path
                .to_string_lossy()
                .contains(digest_key(&second_digest)),
            "second path must carry its source digest"
        );
        assert_eq!(extracts.load(Ordering::Relaxed), 2);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn compiled_directory_artifacts_remain_in_place() {
        let root = temp_root("directory-in-place");
        let source = root.join("fixture.mlmodelc");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("model.mil"), b"compiled model").unwrap();
        let digest = format!("sha256:{}", sha256_directory(&source).unwrap());
        let cache = root.join("cache");
        let materialized = materialize_with_extractor(&source, &digest, &cache, |_, _| {
            bail!("directory artifacts must not invoke extraction")
        })
        .expect("compiled directory stays in place");

        assert_eq!(materialized.path, source);
        assert_eq!(materialized.digest, digest);
        assert!(!cache.exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_or_digest_mismatched_sources_are_refused_before_extraction() {
        let root = temp_root("source-refusal");
        let cache = root.join("cache");
        let missing = root.join("missing.zip");
        let extractor_called = AtomicU64::new(0);
        let missing_result = materialize_with_extractor(
            &missing,
            &format!("sha256:{}", "0".repeat(64)),
            &cache,
            |_, _| {
                extractor_called.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        );
        assert!(missing_result.is_err());

        fs::create_dir_all(&root).unwrap();
        let source = root.join("model.zip");
        fs::write(&source, b"artifact bytes").unwrap();
        let mismatch_result = materialize_with_extractor(
            &source,
            &format!("sha256:{}", "f".repeat(64)),
            &cache,
            |_, _| {
                extractor_called.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        );
        assert!(mismatch_result.is_err());
        assert_eq!(extractor_called.load(Ordering::Relaxed), 0);
        assert!(!cache.join(CACHE_DIRECTORY).exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn system_unzip_materializes_a_compiled_bundle_at_the_digest_path() {
        let root = temp_root("system-unzip");
        let package = root.join("fixture.mlmodelc");
        let archive = root.join("fixture.zip");
        let cache = root.join("cache");
        fs::create_dir_all(&package).unwrap();
        fs::write(package.join("model.mil"), b"compiled-model").unwrap();
        fs::write(package.join("weights.bin"), b"fixture weights").unwrap();
        let status = Command::new("/usr/bin/zip")
            .args(["-q", "-r"])
            .arg(&archive)
            .arg("fixture.mlmodelc")
            .current_dir(&root)
            .status()
            .expect("system zip launches");
        assert!(status.success());
        let digest = file_digest(&archive);

        let materialized = materialize_core_ml_artifact(&archive, &digest, &cache)
            .expect("system unzip materializes the archive");

        assert!(materialized.path.join("model.mil").is_file());
        assert_eq!(
            materialized.path,
            materialized_entry_path(&cache, &digest)
                .unwrap()
                .join("artifact/fixture.mlmodelc")
        );
        assert_eq!(
            materialized.digest,
            format!("sha256:{}", sha256_directory(&materialized.path).unwrap())
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_materializers_publish_one_complete_digest_tree() {
        let root = temp_root("concurrent");
        let source = root.join("model.zip");
        let cache = root.join("cache");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"shared archive bytes").unwrap();
        let digest = file_digest(&source);
        let barrier = Arc::new(Barrier::new(2));
        let outcomes = Arc::new(Mutex::new(Vec::new()));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let source = source.clone();
            let cache = cache.clone();
            let digest = digest.clone();
            let barrier = Arc::clone(&barrier);
            let outcomes = Arc::clone(&outcomes);
            threads.push(std::thread::spawn(move || {
                let outcome =
                    materialize_with_extractor(&source, &digest, &cache, |source, destination| {
                        barrier.wait();
                        write_fake_model(destination, &fs::read(source)?)
                    })
                    .expect("concurrent materialization succeeds");
                outcomes.lock().unwrap().push(outcome);
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }

        let outcomes = outcomes.lock().unwrap();
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].path, outcomes[1].path);
        assert_eq!(outcomes[0].digest, outcomes[1].digest);
        assert!(outcomes.iter().any(|outcome| outcome.reused));
        assert!(outcomes[0].path.join("weights.bin").is_file());
        let cache_entries = fs::read_dir(cache.join(CACHE_DIRECTORY))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(cache_entries, vec![digest_key(&digest).to_string()]);

        drop(outcomes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_extraction_never_becomes_loadable_and_retry_ignores_crash_debris() {
        let root = temp_root("interrupted");
        let source = root.join("model.zip");
        let cache = root.join("cache");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"complete archive bytes").unwrap();
        let digest = file_digest(&source);
        let failed = materialize_with_extractor(&source, &digest, &cache, |_, destination| {
            fs::write(destination.join("model.mil"), b"partial")?;
            bail!("simulated extractor interruption")
        });
        assert!(failed.is_err());
        let final_root = materialized_entry_path(&cache, &digest).unwrap();
        assert!(!final_root.exists());

        let abandoned = cache
            .join(CACHE_DIRECTORY)
            .join(format!(".{}.crashed.tmp", digest_key(&digest)));
        fs::create_dir_all(abandoned.join("artifact")).unwrap();
        fs::write(abandoned.join("artifact/model.mil"), b"partial").unwrap();
        let completed =
            materialize_with_extractor(&source, &digest, &cache, |source, destination| {
                write_fake_model(destination, &fs::read(source)?)
            })
            .expect("retry publishes a complete tree");

        assert!(final_root.is_dir());
        assert_ne!(completed.path, abandoned.join("artifact"));
        assert_eq!(
            fs::read(completed.path.join("weights.bin")).unwrap(),
            b"complete archive bytes"
        );
        assert!(
            abandoned.is_dir(),
            "fresh crash debris is not raced by cleanup"
        );

        fs::remove_dir_all(root).unwrap();
    }

    fn write_fake_model(destination: &Path, payload: &[u8]) -> Result<()> {
        fs::write(destination.join("model.mil"), b"compiled-model")?;
        fs::write(destination.join("weights.bin"), payload)?;
        Ok(())
    }

    fn file_digest(path: &Path) -> String {
        format!("sha256:{}", sha256_file(path).unwrap())
    }

    fn temp_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "synapse-ane-materialization-{label}-{}-{}",
            std::process::id(),
            TEMP_NONCE.fetch_add(1, Ordering::Relaxed)
        ))
    }
}
