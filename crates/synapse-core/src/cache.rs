use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use cortexkit_lease::{FileLeaseStore, LeaseHandle, LeaseKey, LeaseStore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{SanitizedTokenizer, TokenizationError, TokenizerConfig};

const CACHE_MODULE_ID: &str = "models-cache";
const CACHE_LEASE_BACKEND: &str = "file";
const TMP_DIR: &str = "tmp";
const BLOBS_DIR: &str = "blobs";
const META_DIR: &str = "meta";

#[derive(Debug, Error)]
pub enum ModelCacheError {
    #[error("cache io while {action} {path}: {source}")]
    Io {
        action: &'static str,
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("cache json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("cache lease: {0}")]
    Lease(#[from] cortexkit_lease::LeaseError),
    #[error("cache download {url}: {message}")]
    Download { url: String, message: String },
    #[error("artifact_invalid: {0}")]
    ArtifactInvalid(String),
    #[error("cache source must be file://, http://, or https://, got {0}")]
    InvalidSource(String),
    #[error("cache digest {0} is not present")]
    NotFound(String),
    #[error("tokenizer sanitization: {0}")]
    Tokenizer(#[from] TokenizationError),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheValidationState {
    Valid,
    Invalid,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheTombstone {
    pub marked_ms: u64,
    pub marked_by: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCacheMeta {
    pub digest: String,
    pub source_url: String,
    pub format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sanitized_tokenizer_digest: Option<String>,
    pub validation_state: CacheValidationState,
    #[serde(default)]
    pub pins: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tombstone: Option<CacheTombstone>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelCacheIngest {
    pub source_url: String,
    pub expected_digest: Option<String>,
    pub format: String,
    pub tokenizer_path: Option<PathBuf>,
    pub pin_module_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum CacheGcOutcome {
    Deleted {
        digest: String,
    },
    Marked {
        digest: String,
        delete_after_ms: u64,
    },
    Kept {
        digest: String,
        reason: String,
    },
}

#[derive(Debug)]
pub struct ModelCacheReadGuard {
    blob_path: PathBuf,
    _lease: Box<dyn LeaseHandle>,
}

impl ModelCacheReadGuard {
    #[must_use]
    pub fn blob_path(&self) -> &Path {
        &self.blob_path
    }
}

pub struct ModelCache {
    root: PathBuf,
    leases: FileLeaseStore,
}

impl ModelCache {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let leases = FileLeaseStore::new(root.join("leases"));
        Self { root, leases }
    }

    pub fn default_root() -> Result<PathBuf, ModelCacheError> {
        if let Ok(root) = std::env::var("CORTEXKIT_MODEL_CACHE") {
            return Ok(PathBuf::from(root));
        }
        if let Ok(root) = std::env::var("SYNAPSE_MODEL_CACHE_DIR") {
            return Ok(PathBuf::from(root));
        }
        let home = std::env::var_os("HOME").ok_or_else(|| {
            ModelCacheError::InvalidSource("HOME is unset; cannot resolve model cache".to_string())
        })?;
        Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("cortexkit")
            .join("models"))
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn blob_path(&self, digest: &str) -> PathBuf {
        self.root.join(BLOBS_DIR).join(digest_file_name(digest))
    }

    #[must_use]
    pub fn meta_path(&self, digest: &str) -> PathBuf {
        self.root
            .join(META_DIR)
            .join(format!("{}.json", digest_file_name(digest)))
    }

    pub fn acquire_read(&self, digest: &str) -> Result<ModelCacheReadGuard, ModelCacheError> {
        self.ensure_layout()?;
        let normalized = normalize_digest(digest)?;
        let lease = self.leases.acquire_shared(&lease_key(&normalized))?;
        Ok(ModelCacheReadGuard {
            blob_path: self.blob_path(&normalized),
            _lease: lease,
        })
    }

    pub fn ingest(&self, request: ModelCacheIngest) -> Result<ModelCacheMeta, ModelCacheError> {
        self.ensure_layout()?;
        self.cleanup_tmp()?;
        let temp_path = self.root.join(TMP_DIR).join(format!(
            "ingest-{}-{}.tmp",
            std::process::id(),
            now_nanos()
        ));
        let actual_hex = self.write_source_to_temp(&request.source_url, &temp_path)?;
        let actual_digest = format!("sha256:{actual_hex}");
        if let Some(expected) = request.expected_digest.as_deref() {
            let expected = normalize_digest(expected)?;
            if expected != actual_digest {
                let _ = fs::remove_file(&temp_path);
                return Err(ModelCacheError::ArtifactInvalid(format!(
                    "digest mismatch for {}: expected {expected}, got {actual_digest}",
                    request.source_url
                )));
            }
        }

        let blob_path = self.blob_path(&actual_digest);
        if !blob_path.exists() {
            fs::rename(&temp_path, &blob_path).map_err(|source| ModelCacheError::Io {
                action: "publish blob",
                path: blob_path.display().to_string(),
                source,
            })?;
            sync_parent(&blob_path);
        } else {
            fs::remove_file(&temp_path).map_err(|source| ModelCacheError::Io {
                action: "remove duplicate temp blob",
                path: temp_path.display().to_string(),
                source,
            })?;
        }

        let sanitized_tokenizer_digest = request
            .tokenizer_path
            .as_deref()
            .map(|path| {
                SanitizedTokenizer::from_file(
                    path,
                    TokenizerConfig {
                        max_tokens: usize::MAX,
                    },
                )
                .map(|tokenizer| format!("sha256:{}", tokenizer.sanitized_sha256()))
            })
            .transpose()?;

        let mut meta = match self.read_meta(&actual_digest) {
            Ok(mut existing) => {
                existing.source_url = request.source_url;
                existing.format = request.format;
                existing.sanitized_tokenizer_digest = sanitized_tokenizer_digest;
                existing.validation_state = CacheValidationState::Valid;
                existing.tombstone = None;
                existing
            }
            Err(ModelCacheError::NotFound(_)) => ModelCacheMeta {
                digest: actual_digest.clone(),
                source_url: request.source_url,
                format: request.format,
                sanitized_tokenizer_digest,
                validation_state: CacheValidationState::Valid,
                pins: Vec::new(),
                tombstone: None,
            },
            Err(error) => return Err(error),
        };
        if let Some(module_id) = request.pin_module_id {
            add_pin(&mut meta.pins, module_id);
        }
        self.write_meta(&meta)?;
        Ok(meta)
    }

    pub fn pin(&self, digest: &str, module_id: &str) -> Result<ModelCacheMeta, ModelCacheError> {
        self.ensure_layout()?;
        let mut meta = self.read_meta(digest)?;
        add_pin(&mut meta.pins, module_id.to_string());
        meta.tombstone = None;
        self.write_meta(&meta)?;
        Ok(meta)
    }

    pub fn read_meta(&self, digest: &str) -> Result<ModelCacheMeta, ModelCacheError> {
        let normalized = normalize_digest(digest)?;
        let path = self.meta_path(&normalized);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(ModelCacheError::NotFound(normalized));
            }
            Err(source) => {
                return Err(ModelCacheError::Io {
                    action: "read cache metadata",
                    path: path.display().to_string(),
                    source,
                });
            }
        };
        serde_json::from_slice(&bytes).map_err(ModelCacheError::Json)
    }

    pub fn gc_digest(
        &self,
        digest: &str,
        module_id: &str,
        now_ms: u64,
        grace_ms: u64,
    ) -> Result<CacheGcOutcome, ModelCacheError> {
        self.ensure_layout()?;
        let normalized = normalize_digest(digest)?;
        let lease = match self.leases.acquire(&lease_key(&normalized)) {
            Ok(lease) => lease,
            Err(cortexkit_lease::LeaseError::Held { .. }) => {
                return Ok(CacheGcOutcome::Kept {
                    digest: normalized,
                    reason: "shared_reader_or_gc_active".to_string(),
                });
            }
            Err(error) => return Err(error.into()),
        };
        let _lease = lease;

        let mut meta = match self.read_meta(&normalized) {
            Ok(meta) => meta,
            Err(ModelCacheError::NotFound(_)) => {
                return Ok(CacheGcOutcome::Kept {
                    digest: normalized,
                    reason: "missing_metadata".to_string(),
                });
            }
            Err(error) => return Err(error),
        };
        if !meta.pins.is_empty() {
            let reason = if meta.pins.iter().any(|pin| pin == module_id) {
                "pinned_by_requesting_module"
            } else {
                "pinned_by_foreign_module"
            };
            return Ok(CacheGcOutcome::Kept {
                digest: normalized,
                reason: reason.to_string(),
            });
        }

        let marked_ms = match &meta.tombstone {
            Some(tombstone) => tombstone.marked_ms,
            None => {
                meta.tombstone = Some(CacheTombstone {
                    marked_ms: now_ms,
                    marked_by: module_id.to_string(),
                });
                self.write_meta(&meta)?;
                return Ok(CacheGcOutcome::Marked {
                    digest: normalized,
                    delete_after_ms: now_ms.saturating_add(grace_ms),
                });
            }
        };
        let delete_after_ms = marked_ms.saturating_add(grace_ms);
        if now_ms < delete_after_ms {
            return Ok(CacheGcOutcome::Marked {
                digest: normalized,
                delete_after_ms,
            });
        }

        let blob_path = self.blob_path(&normalized);
        match fs::remove_file(&blob_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(ModelCacheError::Io {
                    action: "delete cache blob",
                    path: blob_path.display().to_string(),
                    source,
                });
            }
        }
        let meta_path = self.meta_path(&normalized);
        match fs::remove_file(&meta_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(ModelCacheError::Io {
                    action: "delete cache metadata",
                    path: meta_path.display().to_string(),
                    source,
                });
            }
        }
        sync_parent(&blob_path);
        sync_parent(&meta_path);
        Ok(CacheGcOutcome::Deleted { digest: normalized })
    }

    pub fn gc_all(
        &self,
        module_id: &str,
        now_ms: u64,
        grace_ms: u64,
    ) -> Result<Vec<CacheGcOutcome>, ModelCacheError> {
        self.ensure_layout()?;
        let meta_dir = self.root.join(META_DIR);
        let mut outcomes = Vec::new();
        for entry in fs::read_dir(&meta_dir).map_err(|source| ModelCacheError::Io {
            action: "list cache metadata",
            path: meta_dir.display().to_string(),
            source,
        })? {
            let entry = entry.map_err(|source| ModelCacheError::Io {
                action: "read cache metadata entry",
                path: meta_dir.display().to_string(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            outcomes.push(self.gc_digest(stem, module_id, now_ms, grace_ms)?);
        }
        Ok(outcomes)
    }

    fn ensure_layout(&self) -> Result<(), ModelCacheError> {
        for dir in [
            self.root.join(BLOBS_DIR),
            self.root.join(META_DIR),
            self.root.join(TMP_DIR),
            self.root.join("leases"),
        ] {
            fs::create_dir_all(&dir).map_err(|source| ModelCacheError::Io {
                action: "create cache directory",
                path: dir.display().to_string(),
                source,
            })?;
        }
        Ok(())
    }

    fn cleanup_tmp(&self) -> Result<(), ModelCacheError> {
        let tmp_dir = self.root.join(TMP_DIR);
        if !tmp_dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(&tmp_dir).map_err(|source| ModelCacheError::Io {
            action: "list temp cache directory",
            path: tmp_dir.display().to_string(),
            source,
        })? {
            let entry = entry.map_err(|source| ModelCacheError::Io {
                action: "read temp cache entry",
                path: tmp_dir.display().to_string(),
                source,
            })?;
            let path = entry.path();
            if path.is_file() {
                fs::remove_file(&path).map_err(|source| ModelCacheError::Io {
                    action: "remove stale temp cache file",
                    path: path.display().to_string(),
                    source,
                })?;
            }
        }
        Ok(())
    }

    fn write_source_to_temp(
        &self,
        source_url: &str,
        temp_path: &Path,
    ) -> Result<String, ModelCacheError> {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(temp_path)
            .map_err(|source| ModelCacheError::Io {
                action: "create temp cache blob",
                path: temp_path.display().to_string(),
                source,
            })?;
        let mut hasher = Sha256::new();
        if let Some(path) = source_url.strip_prefix("file://") {
            let mut input = File::open(path).map_err(|source| ModelCacheError::Io {
                action: "open source artifact",
                path: path.to_string(),
                source,
            })?;
            copy_and_hash(&mut input, &mut output, &mut hasher, temp_path)?;
        } else if source_url.starts_with("http://") || source_url.starts_with("https://") {
            let mut response = reqwest::blocking::get(source_url)
                .map_err(|error| ModelCacheError::Download {
                    url: source_url.to_string(),
                    message: error.to_string(),
                })?
                .error_for_status()
                .map_err(|error| ModelCacheError::Download {
                    url: source_url.to_string(),
                    message: error.to_string(),
                })?;
            copy_and_hash(&mut response, &mut output, &mut hasher, temp_path)?;
        } else if !source_url.contains("://") {
            let mut input = File::open(source_url).map_err(|source| ModelCacheError::Io {
                action: "open source artifact",
                path: source_url.to_string(),
                source,
            })?;
            copy_and_hash(&mut input, &mut output, &mut hasher, temp_path)?;
        } else {
            let _ = fs::remove_file(temp_path);
            return Err(ModelCacheError::InvalidSource(source_url.to_string()));
        }
        output.sync_all().map_err(|source| ModelCacheError::Io {
            action: "fsync temp cache blob",
            path: temp_path.display().to_string(),
            source,
        })?;
        sync_parent(temp_path);
        Ok(hex::encode(hasher.finalize()))
    }

    fn write_meta(&self, meta: &ModelCacheMeta) -> Result<(), ModelCacheError> {
        let path = self.meta_path(&meta.digest);
        let temp_path = self.root.join(TMP_DIR).join(format!(
            "meta-{}-{}.tmp",
            digest_file_name(&meta.digest),
            now_nanos()
        ));
        let bytes = serde_json::to_vec_pretty(meta)?;
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
                .map_err(|source| ModelCacheError::Io {
                    action: "create temp cache metadata",
                    path: temp_path.display().to_string(),
                    source,
                })?;
            file.write_all(&bytes)
                .map_err(|source| ModelCacheError::Io {
                    action: "write temp cache metadata",
                    path: temp_path.display().to_string(),
                    source,
                })?;
            file.sync_all().map_err(|source| ModelCacheError::Io {
                action: "fsync temp cache metadata",
                path: temp_path.display().to_string(),
                source,
            })?;
        }
        fs::rename(&temp_path, &path).map_err(|source| ModelCacheError::Io {
            action: "publish cache metadata",
            path: path.display().to_string(),
            source,
        })?;
        sync_parent(&path);
        Ok(())
    }
}

fn add_pin(pins: &mut Vec<String>, module_id: String) {
    if !pins.iter().any(|pin| pin == &module_id) {
        pins.push(module_id);
        pins.sort();
    }
}

fn copy_and_hash(
    input: &mut impl Read,
    output: &mut File,
    hasher: &mut Sha256,
    temp_path: &Path,
) -> Result<(), ModelCacheError> {
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|source| ModelCacheError::Io {
                action: "read source artifact",
                path: temp_path.display().to_string(),
                source,
            })?;
        if read == 0 {
            return Ok(());
        }
        hasher.update(&buffer[..read]);
        output
            .write_all(&buffer[..read])
            .map_err(|source| ModelCacheError::Io {
                action: "write temp cache blob",
                path: temp_path.display().to_string(),
                source,
            })?;
    }
}

fn normalize_digest(value: &str) -> Result<String, ModelCacheError> {
    let raw = value.strip_prefix("sha256:").unwrap_or(value).trim();
    if raw.len() != 64 || !raw.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ModelCacheError::ArtifactInvalid(format!(
            "sha256 digest must be 64 hex characters, got '{value}'"
        )));
    }
    Ok(format!("sha256:{}", raw.to_ascii_lowercase()))
}

fn digest_file_name(digest: &str) -> String {
    digest.strip_prefix("sha256:").unwrap_or(digest).to_string()
}

fn lease_key(digest: &str) -> LeaseKey {
    LeaseKey::new(
        CACHE_MODULE_ID,
        CACHE_LEASE_BACKEND,
        normalize_for_lease(digest),
    )
}

fn normalize_for_lease(digest: &str) -> String {
    digest.strip_prefix("sha256:").unwrap_or(digest).to_string()
}

fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_rejects_digest_mismatch_without_publishing_blob() {
        let root = temp_root("digest-mismatch");
        let cache = ModelCache::new(&root);
        let source = root.join("source.bin");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"artifact-a").unwrap();

        let error = cache
            .ingest(ModelCacheIngest {
                source_url: format!("file://{}", source.display()),
                expected_digest: Some(format!("sha256:{}", "0".repeat(64))),
                format: "bin".to_string(),
                tokenizer_path: None,
                pin_module_id: None,
            })
            .expect_err("mismatched digest should reject");
        assert!(matches!(error, ModelCacheError::ArtifactInvalid(_)));
        assert!(fs::read_dir(root.join(BLOBS_DIR)).unwrap().next().is_none());
    }

    #[test]
    fn ingest_cleans_stale_temp_and_publishes_atomically() {
        let root = temp_root("crash-safe-ingest");
        let tmp = root.join(TMP_DIR);
        fs::create_dir_all(&tmp).unwrap();
        let stale = tmp.join("left-behind.tmp");
        fs::write(&stale, b"partial").unwrap();
        let source = root.join("source.bin");
        fs::write(&source, b"artifact-b").unwrap();

        let cache = ModelCache::new(&root);
        let meta = cache
            .ingest(ModelCacheIngest {
                source_url: format!("file://{}", source.display()),
                expected_digest: None,
                format: "bin".to_string(),
                tokenizer_path: None,
                pin_module_id: None,
            })
            .expect("ingest should publish final blob");

        assert!(!stale.exists(), "orphaned temp files should be removed");
        assert!(cache.blob_path(&meta.digest).exists());
        assert!(cache.meta_path(&meta.digest).exists());
    }

    #[test]
    fn pin_protects_blob_from_gc() {
        let (cache, meta) = ingest_fixture("pin-protect", b"artifact-c", Some("synapse"));

        let outcome = cache
            .gc_digest(&meta.digest, "synapse", 10, 0)
            .expect("gc should inspect pinned metadata");

        assert_eq!(
            outcome,
            CacheGcOutcome::Kept {
                digest: meta.digest.clone(),
                reason: "pinned_by_requesting_module".to_string(),
            }
        );
        assert!(cache.blob_path(&meta.digest).exists());
    }

    #[test]
    fn shared_reader_blocks_gc_delete_until_released() {
        let (cache, meta) = ingest_fixture("reader-gc", b"artifact-d", None);
        let read_guard = cache
            .acquire_read(&meta.digest)
            .expect("reader should acquire shared lease");

        let kept = cache
            .gc_digest(&meta.digest, "synapse", 10, 0)
            .expect("gc should observe held shared lease");
        assert_eq!(
            kept,
            CacheGcOutcome::Kept {
                digest: meta.digest.clone(),
                reason: "shared_reader_or_gc_active".to_string(),
            }
        );
        assert!(read_guard.blob_path().exists());
        drop(read_guard);

        assert!(matches!(
            cache.gc_digest(&meta.digest, "synapse", 20, 5).unwrap(),
            CacheGcOutcome::Marked { .. }
        ));
        assert!(cache.blob_path(&meta.digest).exists());
        assert_eq!(
            cache.gc_digest(&meta.digest, "synapse", 25, 5).unwrap(),
            CacheGcOutcome::Deleted {
                digest: meta.digest.clone()
            }
        );
        assert!(!cache.blob_path(&meta.digest).exists());
    }

    fn ingest_fixture(
        label: &str,
        contents: &[u8],
        pin: Option<&str>,
    ) -> (ModelCache, ModelCacheMeta) {
        let root = temp_root(label);
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.bin");
        fs::write(&source, contents).unwrap();
        let cache = ModelCache::new(&root);
        let meta = cache
            .ingest(ModelCacheIngest {
                source_url: format!("file://{}", source.display()),
                expected_digest: None,
                format: "bin".to_string(),
                tokenizer_path: None,
                pin_module_id: pin.map(str::to_string),
            })
            .expect("fixture ingest");
        (cache, meta)
    }

    fn temp_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "synapse-cache-{label}-{}-{}",
            std::process::id(),
            now_nanos()
        ))
    }
}
