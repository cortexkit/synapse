//! The release-versioned model catalog: the curated embed and rerank models a
//! user can browse, download and serve by catalog id.
//!
//! The catalog is one JSON document, `models.json`, compiled into the module.
//! Each entry pins an upstream Hugging Face repository at a commit, lists the
//! files to fetch with their sha256 and size, and declares the backends that
//! serve it with the exact load parameters and the fingerprint the module
//! mints when it loads those files.
//!
//! Validation has two layers:
//! - [`validate_schema`] holds for any catalog document, including the
//!   alternate catalogs tests use, and checks the structural invariants;
//! - [`validate_release`] holds only for the compiled `models.json` and
//!   additionally pins the frozen ids, revisions, backend sets and load
//!   literals of this release, so a careless edit cannot ship a model or
//!   backend that was never verified.

// The download, serving and release-check code that reads this module is
// built on top of it separately; until all of it is wired in, a non-test
// build sees some of these items unused.
#![cfg_attr(not(test), allow(dead_code))]

pub(crate) mod evidence;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The compiled catalog document shipped with this release.
pub(crate) const COMPILED_CATALOG_JSON: &str = include_str!("models.json");

/// Every backend value a catalog entry may declare. `cpu`, `ort` and `llama`
/// are deliberately absent: embedding and reranking serve only on owned
/// accelerated lanes.
pub(crate) const ALLOWED_BACKENDS: [&str; 4] = ["metal", "ane", "cuda", "vulkan"];

/// Every engine value a catalog backend may name in this release.
pub(crate) const ALLOWED_ENGINES: [&str; 1] = ["owned-metal"];

/// Every file role. Per backend there is exactly one `model` and one
/// `tokenizer` file and at most one `config` file.
pub(crate) const ALLOWED_ROLES: [&str; 3] = ["model", "tokenizer", "config"];

/// Pooling values the worker protocol parses (`WorkerPooling`).
const ALLOWED_POOLING: [&str; 3] = ["mean", "cls", "last"];

const EMBED_SELF_CHECK_INPUTS: usize = 8;
const RERANK_SELF_CHECK_INPUTS: usize = 4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Catalog {
    pub catalog_revision: i64,
    pub models: Vec<CatalogEntry>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CatalogEntry {
    pub id: String,
    pub task: String,
    pub name: String,
    pub description: String,
    pub default_for_task: bool,
    pub upstream: Upstream,
    pub files: Vec<CatalogFile>,
    pub backends: Vec<CatalogBackend>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub self_check: Option<SelfCheck>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Upstream {
    pub hf_repo: String,
    pub revision: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CatalogFile {
    pub path: String,
    pub role: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub backends: Vec<String>,
}

/// One declared backend of an entry. The load-parameter fields are optional
/// in the type so that a document missing one fails schema validation with a
/// named violation instead of an opaque parse error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CatalogBackend {
    pub backend: String,
    pub engine: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_units: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pooling: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dims: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_abs_tolerance: Option<f64>,
    pub fingerprint: String,
}

/// The reference data a freshly loaded catalog lane is checked against on
/// the user's machine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SelfCheck {
    pub fixture_revision: i64,
    pub reference_tool: String,
    pub inputs: Vec<SelfCheckInput>,
    pub reference: SelfCheckReference,
}

/// An embed self-check input is a plain string; a rerank input is a query
/// with its candidates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum SelfCheckInput {
    Text(String),
    Rerank(RerankSelfCheckInput),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RerankSelfCheckInput {
    pub query: String,
    pub candidates: Vec<String>,
}

/// `vectors` (embed, one per input) or `scores` (rerank, `sigmoid(raw
/// logit)` per candidate), never both.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SelfCheckReference {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vectors: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scores: Option<Vec<Vec<f64>>>,
}

/// One violated catalog invariant. Schema violations name the entry (and
/// backend or file) at fault; release violations name the frozen value the
/// compiled catalog departed from.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub(crate) enum CatalogError {
    #[error("catalog document is not valid catalog JSON: {0}")]
    Malformed(String),
    #[error("catalog_revision must be a positive integer, found {0}")]
    NonPositiveCatalogRevision(i64),
    #[error("catalog entry id '{0}' appears more than once")]
    DuplicateEntryId(String),
    #[error("catalog entry id '{0}' must be non-empty lowercase [a-z0-9._-]")]
    InvalidEntryId(String),
    #[error("catalog entry id '{id}' collides with a lane id derived from entry '{owner}'")]
    ReservedIdCollision { id: String, owner: String },
    #[error("catalog entry '{id}' has task '{task}', expected embed or rerank")]
    UnknownTask { id: String, task: String },
    #[error("catalog entry '{id}' has hf_repo '{hf_repo}', expected <owner>/<name>")]
    InvalidRepo { id: String, hf_repo: String },
    #[error("catalog entry '{id}' has revision '{revision}', expected a 40-hex commit")]
    InvalidRevision { id: String, revision: String },
    #[error("task '{task}' has {count} default_for_task entries, expected exactly one")]
    DefaultForTask { task: String, count: usize },
    #[error("embed entry '{id}' mentions rerank in its id, name or description")]
    RerankWordInEmbedEntry { id: String },
    #[error("catalog entry '{id}' declares backend '{backend}' more than once")]
    DuplicateBackend { id: String, backend: String },
    #[error("catalog entry '{id}' declares backend '{backend}', expected one of metal, ane, cuda, vulkan")]
    UnknownBackend { id: String, backend: String },
    #[error(
        "catalog entry '{id}' backend '{backend}' names engine '{engine}', which is not allowed"
    )]
    UnknownEngine {
        id: String,
        backend: String,
        engine: String,
    },
    #[error("catalog entry '{id}' backend '{backend}' is missing {field}")]
    MissingBackendField {
        id: String,
        backend: String,
        field: &'static str,
    },
    #[error("catalog entry '{id}' backend '{backend}' has invalid {field}: {value}")]
    InvalidBackendField {
        id: String,
        backend: String,
        field: &'static str,
        value: String,
    },
    #[error("catalog entry '{id}' backend '{backend}' has attention_units {attention_units} below max_tokens² for max_tokens {max_tokens}")]
    AttentionUnitsBelowContext {
        id: String,
        backend: String,
        attention_units: u64,
        max_tokens: u64,
    },
    #[error("rerank entry '{id}' backend '{backend}' declares embed-only field {field}")]
    EmbedFieldOnRerank {
        id: String,
        backend: String,
        field: &'static str,
    },
    #[error(
        "embed entry '{id}' backend '{backend}' declares rerank-only field rerank_abs_tolerance"
    )]
    RerankFieldOnEmbed { id: String, backend: String },
    #[error("catalog entry '{id}' backend '{backend}' has fingerprint '{fingerprint}', expected 64 lowercase hex")]
    InvalidFingerprint {
        id: String,
        backend: String,
        fingerprint: String,
    },
    #[error("catalog entry '{id}' lists file path '{path}' more than once")]
    DuplicateFilePath { id: String, path: String },
    #[error(
        "catalog entry '{id}' file '{path}' has role '{role}', expected model, tokenizer or config"
    )]
    UnknownRole {
        id: String,
        path: String,
        role: String,
    },
    #[error("catalog entry '{id}' file '{path}' has sha256 '{sha256}', expected 64 lowercase hex")]
    InvalidSha256 {
        id: String,
        path: String,
        sha256: String,
    },
    #[error("catalog entry '{id}' file '{path}' lists no backends")]
    FileBackendsEmpty { id: String, path: String },
    #[error("catalog entry '{id}' file '{path}' names backend '{backend}', which the entry does not declare")]
    FileBackendUndeclared {
        id: String,
        path: String,
        backend: String,
    },
    #[error("catalog entry '{id}' backend '{backend}' has {count} '{role}' files")]
    RoleCardinality {
        id: String,
        backend: String,
        role: &'static str,
        count: usize,
    },
    #[error("catalog entry '{id}' declares backends but no self_check")]
    SelfCheckMissing { id: String },
    #[error("catalog entry '{id}' declares no backends but has a self_check")]
    SelfCheckUnexpected { id: String },
    #[error("catalog entry '{id}' self_check is invalid: {reason}")]
    SelfCheckShape { id: String, reason: String },
    #[error("compiled catalog ids are {actual:?}, the frozen set is {expected:?}")]
    FrozenIdSet {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    #[error("compiled catalog entry '{id}' has {field} {actual}, the frozen value is {expected}")]
    FrozenMismatch {
        id: String,
        field: String,
        expected: String,
        actual: String,
    },
}

/// Parses a catalog document without validating it.
pub(crate) fn parse_catalog(json: &str) -> Result<Catalog, CatalogError> {
    serde_json::from_str(json).map_err(|error| CatalogError::Malformed(error.to_string()))
}

/// Parses and schema-validates a catalog document, such as a test fixture
/// catalog. Release validation does not apply.
pub(crate) fn load_schema_valid(json: &str) -> Result<Catalog, CatalogError> {
    let catalog = parse_catalog(json)?;
    validate_schema(&catalog)?;
    Ok(catalog)
}

/// Parses and release-validates a catalog document as the compiled catalog.
pub(crate) fn load_release_valid(json: &str) -> Result<Catalog, CatalogError> {
    let catalog = parse_catalog(json)?;
    validate_release(&catalog)?;
    Ok(catalog)
}

/// The compiled catalog, parsed and release-validated once. A unit test
/// keeps it valid, so an error here means the binary was built from a
/// catalog that never passed `cargo test`.
pub(crate) fn compiled_catalog() -> Result<&'static Catalog, &'static CatalogError> {
    static COMPILED: OnceLock<Result<Catalog, CatalogError>> = OnceLock::new();
    COMPILED
        .get_or_init(|| load_release_valid(COMPILED_CATALOG_JSON))
        .as_ref()
}

impl Catalog {
    pub(crate) fn entry(&self, id: &str) -> Option<&CatalogEntry> {
        self.models.iter().find(|entry| entry.id == id)
    }

    /// Resolves a catalog-reserved id: a catalog id yields its entry with no
    /// backend, and `<catalog_id>-<backend>` for any allowed backend value
    /// yields the entry and that backend, declared or not. Returns `None`
    /// for ids the catalog does not reserve.
    pub(crate) fn resolve_reserved(
        &self,
        id: &str,
    ) -> Option<(&CatalogEntry, Option<&'static str>)> {
        if let Some(entry) = self.entry(id) {
            return Some((entry, None));
        }
        ALLOWED_BACKENDS.iter().find_map(|backend| {
            let catalog_id = id.strip_suffix(backend)?.strip_suffix('-')?;
            self.entry(catalog_id).map(|entry| (entry, Some(*backend)))
        })
    }

    /// True for every catalog id and every `<catalog_id>-<backend>` over the
    /// allowed backend values, which free-form registrations may not use.
    pub(crate) fn is_reserved_id(&self, id: &str) -> bool {
        self.resolve_reserved(id).is_some()
    }
}

impl CatalogEntry {
    pub(crate) fn backend(&self, backend: &str) -> Option<&CatalogBackend> {
        self.backends.iter().find(|row| row.backend == backend)
    }

    /// The lowercase hex sha256 of the JCS serialization of
    /// `{upstream, files}`: the identity of the bytes this entry installs.
    pub(crate) fn manifest_digest(&self) -> String {
        let manifest = serde_json::json!({
            "upstream": &self.upstream,
            "files": &self.files,
        });
        let canonical = jcs(&manifest)
            .expect("catalog manifests hold only strings, integers, and arrays/objects of them");
        hex::encode(Sha256::digest(canonical.as_bytes()))
    }

    /// The sum of `size_bytes` over this entry's distinct files serving at
    /// least one of `runnable` backends; 0 when none is runnable here.
    pub(crate) fn download_bytes(&self, runnable: &[&str]) -> u64 {
        self.files
            .iter()
            .filter(|file| {
                file.backends
                    .iter()
                    .any(|backend| runnable.contains(&backend.as_str()))
            })
            .map(|file| file.size_bytes)
            .sum()
    }

    /// The files a backend loads, keyed by role.
    pub(crate) fn backend_files(&self, backend: &str) -> BTreeMap<&str, &CatalogFile> {
        self.files
            .iter()
            .filter(|file| file.backends.iter().any(|name| name == backend))
            .map(|file| (file.role.as_str(), file))
            .collect()
    }
}

/// The runtime lane id of one declared backend of a catalog entry.
pub(crate) fn lane_id(catalog_id: &str, backend: &str) -> String {
    format!("{catalog_id}-{backend}")
}

/// RFC 8785 (JCS) canonical serialization for the JSON this module hashes.
///
/// Object members are sorted by the UTF-16 code units of their names,
/// there is no insignificant whitespace, and strings use the minimal JSON
/// escaping RFC 8785 prescribes (which is serde_json's). Numbers are limited
/// to integers of magnitude at most 2^53 - 1, which print identically under
/// RFC 8785's ECMAScript number rule; anything else is refused rather than
/// risk a digest that another JCS implementation would not reproduce.
pub(crate) fn jcs(value: &Value) -> Result<String, String> {
    let mut out = String::new();
    write_jcs(value, &mut out)?;
    Ok(out)
}

const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

fn write_jcs(value: &Value, out: &mut String) -> Result<(), String> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => {
            if let Some(unsigned) = number.as_u64().filter(|n| *n <= MAX_SAFE_INTEGER) {
                out.push_str(&unsigned.to_string());
            } else if let Some(signed) = number
                .as_i64()
                .filter(|n| n.unsigned_abs() <= MAX_SAFE_INTEGER)
            {
                out.push_str(&signed.to_string());
            } else {
                return Err(format!(
                    "JCS serialization here supports only integers within ±(2^53 - 1), found {number}"
                ));
            }
        }
        Value::String(text) => {
            out.push_str(&serde_json::to_string(text).map_err(|e| e.to_string())?)
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_jcs(item, out)?;
            }
            out.push(']');
        }
        Value::Object(members) => {
            let mut sorted: Vec<(&String, &Value)> = members.iter().collect();
            sorted.sort_by(|(left, _), (right, _)| left.encode_utf16().cmp(right.encode_utf16()));
            out.push('{');
            for (index, (name, member)) in sorted.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(name).map_err(|e| e.to_string())?);
                out.push(':');
                write_jcs(member, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_valid_entry_id(id: &str) -> bool {
    !id.is_empty()
        && id.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

/// Checks the structural invariants every catalog document must satisfy,
/// returning the first violation found.
pub(crate) fn validate_schema(catalog: &Catalog) -> Result<(), CatalogError> {
    if catalog.catalog_revision <= 0 {
        return Err(CatalogError::NonPositiveCatalogRevision(
            catalog.catalog_revision,
        ));
    }
    let mut ids = BTreeSet::new();
    for entry in &catalog.models {
        if !is_valid_entry_id(&entry.id) {
            return Err(CatalogError::InvalidEntryId(entry.id.clone()));
        }
        if !ids.insert(entry.id.as_str()) {
            return Err(CatalogError::DuplicateEntryId(entry.id.clone()));
        }
    }
    // A catalog id must not double as another entry's derived lane id, or a
    // reserved id would resolve two ways.
    for entry in &catalog.models {
        for owner in &catalog.models {
            if ALLOWED_BACKENDS
                .iter()
                .any(|backend| entry.id == lane_id(&owner.id, backend))
            {
                return Err(CatalogError::ReservedIdCollision {
                    id: entry.id.clone(),
                    owner: owner.id.clone(),
                });
            }
        }
    }
    for entry in &catalog.models {
        validate_entry(entry)?;
    }
    for task in ["embed", "rerank"] {
        let count = catalog
            .models
            .iter()
            .filter(|entry| entry.task == task && entry.default_for_task)
            .count();
        if count != 1 {
            return Err(CatalogError::DefaultForTask {
                task: task.to_string(),
                count,
            });
        }
    }
    Ok(())
}

fn validate_entry(entry: &CatalogEntry) -> Result<(), CatalogError> {
    let id = entry.id.clone();
    let embed = match entry.task.as_str() {
        "embed" => true,
        "rerank" => false,
        _ => {
            return Err(CatalogError::UnknownTask {
                id,
                task: entry.task.clone(),
            })
        }
    };
    if embed
        && [&entry.id, &entry.name, &entry.description]
            .iter()
            .any(|text| text.to_ascii_lowercase().contains("rerank"))
    {
        return Err(CatalogError::RerankWordInEmbedEntry { id });
    }
    let repo_ok = entry
        .upstream
        .hf_repo
        .split_once('/')
        .is_some_and(|(owner, name)| !owner.is_empty() && !name.is_empty() && !name.contains('/'));
    if !repo_ok {
        return Err(CatalogError::InvalidRepo {
            id,
            hf_repo: entry.upstream.hf_repo.clone(),
        });
    }
    if !is_lower_hex(&entry.upstream.revision, 40) {
        return Err(CatalogError::InvalidRevision {
            id,
            revision: entry.upstream.revision.clone(),
        });
    }

    let mut declared = BTreeSet::new();
    for backend in &entry.backends {
        if !ALLOWED_BACKENDS.contains(&backend.backend.as_str()) {
            return Err(CatalogError::UnknownBackend {
                id,
                backend: backend.backend.clone(),
            });
        }
        if !declared.insert(backend.backend.as_str()) {
            return Err(CatalogError::DuplicateBackend {
                id,
                backend: backend.backend.clone(),
            });
        }
        validate_backend(&entry.id, embed, backend)?;
    }

    let mut paths = BTreeSet::new();
    for file in &entry.files {
        if !paths.insert(file.path.as_str()) {
            return Err(CatalogError::DuplicateFilePath {
                id,
                path: file.path.clone(),
            });
        }
        if !ALLOWED_ROLES.contains(&file.role.as_str()) {
            return Err(CatalogError::UnknownRole {
                id,
                path: file.path.clone(),
                role: file.role.clone(),
            });
        }
        if !is_lower_hex(&file.sha256, 64) {
            return Err(CatalogError::InvalidSha256 {
                id,
                path: file.path.clone(),
                sha256: file.sha256.clone(),
            });
        }
        if file.backends.is_empty() {
            return Err(CatalogError::FileBackendsEmpty {
                id,
                path: file.path.clone(),
            });
        }
        if let Some(undeclared) = file
            .backends
            .iter()
            .find(|backend| !declared.contains(backend.as_str()))
        {
            return Err(CatalogError::FileBackendUndeclared {
                id,
                path: file.path.clone(),
                backend: undeclared.clone(),
            });
        }
    }
    for backend in &entry.backends {
        let files = entry
            .files
            .iter()
            .filter(|file| file.backends.contains(&backend.backend));
        for (role, min, max) in [("model", 1, 1), ("tokenizer", 1, 1), ("config", 0, 1)] {
            let count = files.clone().filter(|file| file.role == role).count();
            if count < min || count > max {
                return Err(CatalogError::RoleCardinality {
                    id,
                    backend: backend.backend.clone(),
                    role,
                    count,
                });
            }
        }
    }

    match (&entry.self_check, entry.backends.is_empty()) {
        (None, false) => Err(CatalogError::SelfCheckMissing { id }),
        (Some(_), true) => Err(CatalogError::SelfCheckUnexpected { id }),
        (None, true) => Ok(()),
        (Some(check), false) => validate_self_check(entry, embed, check)
            .map_err(|reason| CatalogError::SelfCheckShape { id, reason }),
    }
}

fn validate_backend(id: &str, embed: bool, row: &CatalogBackend) -> Result<(), CatalogError> {
    let backend = row.backend.clone();
    if !ALLOWED_ENGINES.contains(&row.engine.as_str()) {
        return Err(CatalogError::UnknownEngine {
            id: id.to_string(),
            backend,
            engine: row.engine.clone(),
        });
    }
    let missing = |field: &'static str| CatalogError::MissingBackendField {
        id: id.to_string(),
        backend: backend.clone(),
        field,
    };
    let invalid = |field: &'static str, value: String| CatalogError::InvalidBackendField {
        id: id.to_string(),
        backend: backend.clone(),
        field,
        value,
    };
    for (field, value) in [
        ("family", &row.family),
        ("dtype", &row.dtype),
        ("execution", &row.execution),
    ] {
        match value {
            None => return Err(missing(field)),
            Some(text) if text.trim().is_empty() => return Err(invalid(field, text.clone())),
            Some(_) => {}
        }
    }
    let attention_units = row
        .attention_units
        .ok_or_else(|| missing("attention_units"))?;
    let max_tokens = row.max_tokens.ok_or_else(|| missing("max_tokens"))?;
    if max_tokens == 0 {
        return Err(invalid("max_tokens", "0".to_string()));
    }
    if u128::from(attention_units) < u128::from(max_tokens) * u128::from(max_tokens) {
        return Err(CatalogError::AttentionUnitsBelowContext {
            id: id.to_string(),
            backend,
            attention_units,
            max_tokens,
        });
    }
    if embed {
        let pooling = row.pooling.as_ref().ok_or_else(|| missing("pooling"))?;
        if !ALLOWED_POOLING.contains(&pooling.as_str()) {
            return Err(invalid("pooling", pooling.clone()));
        }
        row.normalize.ok_or_else(|| missing("normalize"))?;
        match row.dims {
            None => return Err(missing("dims")),
            Some(0) => return Err(invalid("dims", "0".to_string())),
            Some(_) => {}
        }
        if row.rerank_abs_tolerance.is_some() {
            return Err(CatalogError::RerankFieldOnEmbed {
                id: id.to_string(),
                backend,
            });
        }
    } else {
        for (field, present) in [
            ("pooling", row.pooling.is_some()),
            ("normalize", row.normalize.is_some()),
            ("dims", row.dims.is_some()),
        ] {
            if present {
                return Err(CatalogError::EmbedFieldOnRerank {
                    id: id.to_string(),
                    backend,
                    field,
                });
            }
        }
        match row.rerank_abs_tolerance {
            None => return Err(missing("rerank_abs_tolerance")),
            Some(tolerance) if !(tolerance.is_finite() && tolerance > 0.0) => {
                return Err(invalid("rerank_abs_tolerance", tolerance.to_string()))
            }
            Some(_) => {}
        }
    }
    if !is_lower_hex(&row.fingerprint, 64) {
        return Err(CatalogError::InvalidFingerprint {
            id: id.to_string(),
            backend,
            fingerprint: row.fingerprint.clone(),
        });
    }
    Ok(())
}

fn validate_self_check(entry: &CatalogEntry, embed: bool, check: &SelfCheck) -> Result<(), String> {
    if check.fixture_revision <= 0 {
        return Err(format!(
            "fixture_revision must be a positive integer, found {}",
            check.fixture_revision
        ));
    }
    if check.reference_tool.trim().is_empty() {
        return Err("reference_tool is empty".to_string());
    }
    if embed {
        if check.inputs.len() != EMBED_SELF_CHECK_INPUTS {
            return Err(format!(
                "embed self_check needs {EMBED_SELF_CHECK_INPUTS} inputs, found {}",
                check.inputs.len()
            ));
        }
        if !check
            .inputs
            .iter()
            .all(|input| matches!(input, SelfCheckInput::Text(_)))
        {
            return Err("embed self_check inputs must be strings".to_string());
        }
        if check.reference.scores.is_some() {
            return Err("embed self_check reference must not carry scores".to_string());
        }
        let vectors = check
            .reference
            .vectors
            .as_ref()
            .ok_or("embed self_check reference has no vectors")?;
        if vectors.len() != check.inputs.len() {
            return Err(format!(
                "embed self_check has {} inputs but {} reference vectors",
                check.inputs.len(),
                vectors.len()
            ));
        }
        // Every embed backend declares dims; the vectors must match each.
        for backend in &entry.backends {
            let dims = backend.dims.unwrap_or_default();
            if let Some((index, vector)) = vectors
                .iter()
                .enumerate()
                .find(|(_, vector)| vector.len() as u64 != dims)
            {
                return Err(format!(
                    "reference vector {index} has {} values, backend '{}' declares dims {dims}",
                    vector.len(),
                    backend.backend
                ));
            }
        }
        if vectors.iter().flatten().any(|value| !value.is_finite()) {
            return Err("reference vectors must be finite".to_string());
        }
    } else {
        if check.inputs.len() != RERANK_SELF_CHECK_INPUTS {
            return Err(format!(
                "rerank self_check needs {RERANK_SELF_CHECK_INPUTS} inputs, found {}",
                check.inputs.len()
            ));
        }
        if check.reference.vectors.is_some() {
            return Err("rerank self_check reference must not carry vectors".to_string());
        }
        let scores = check
            .reference
            .scores
            .as_ref()
            .ok_or("rerank self_check reference has no scores")?;
        if scores.len() != check.inputs.len() {
            return Err(format!(
                "rerank self_check has {} inputs but {} score rows",
                check.inputs.len(),
                scores.len()
            ));
        }
        for (index, (input, row)) in check.inputs.iter().zip(scores).enumerate() {
            let SelfCheckInput::Rerank(input) = input else {
                return Err("rerank self_check inputs must be {query, candidates}".to_string());
            };
            if input.candidates.is_empty() {
                return Err(format!("rerank self_check input {index} has no candidates"));
            }
            if row.len() != input.candidates.len() {
                return Err(format!(
                    "rerank self_check input {index} has {} candidates but {} scores",
                    input.candidates.len(),
                    row.len()
                ));
            }
            // Scores are sigmoid(raw logit), so they lie in [0, 1].
            if row
                .iter()
                .any(|score| !(score.is_finite() && (0.0..=1.0).contains(score)))
            {
                return Err(format!(
                    "rerank self_check scores for input {index} must be finite and within [0, 1]"
                ));
            }
        }
    }
    Ok(())
}

/// The frozen load literals of one declared backend in this release.
struct FrozenBackend {
    backend: &'static str,
    engine: &'static str,
    family: &'static str,
    dtype: &'static str,
    execution: &'static str,
    attention_units: u64,
    max_tokens: u64,
    pooling: Option<&'static str>,
    normalize: Option<bool>,
    dims: Option<u64>,
    rerank_abs_tolerance: Option<f64>,
}

/// The frozen identity of one compiled catalog entry in this release.
struct FrozenEntry {
    id: &'static str,
    hf_repo: &'static str,
    revision: &'static str,
    default_for_task: bool,
    backends: &'static [FrozenBackend],
}

/// `max_tokens` 8192 needs `attention_units >= 8192²`.
const FROZEN_ATTENTION_UNITS: u64 = 67_108_864;
const FROZEN_MAX_TOKENS: u64 = 8192;

/// What this release ships. Changing any value here is a release decision
/// that needs new evidence records for the affected backends.
const FROZEN_ENTRIES: [FrozenEntry; 4] = [
    FrozenEntry {
        id: "gte-modernbert-base",
        hf_repo: "Alibaba-NLP/gte-modernbert-base",
        revision: "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        default_for_task: true,
        backends: &[FrozenBackend {
            backend: "metal",
            engine: "owned-metal",
            family: "gte-modernbert",
            dtype: "f16",
            execution: "explicit",
            attention_units: FROZEN_ATTENTION_UNITS,
            max_tokens: FROZEN_MAX_TOKENS,
            pooling: Some("cls"),
            normalize: Some(true),
            dims: Some(768),
            rerank_abs_tolerance: None,
        }],
    },
    FrozenEntry {
        id: "gte-reranker-modernbert-base",
        hf_repo: "Alibaba-NLP/gte-reranker-modernbert-base",
        revision: "f7481e6055501a30fb19d090657df9ec1f79ab2c",
        default_for_task: true,
        backends: &[FrozenBackend {
            backend: "metal",
            engine: "owned-metal",
            family: "gte-modernbert",
            dtype: "f32",
            execution: "explicit",
            attention_units: FROZEN_ATTENTION_UNITS,
            max_tokens: FROZEN_MAX_TOKENS,
            pooling: None,
            normalize: None,
            dims: None,
            rerank_abs_tolerance: Some(0.005),
        }],
    },
    FrozenEntry {
        id: "qwen3-embedding-0.6b",
        hf_repo: "Qwen/Qwen3-Embedding-0.6B",
        revision: "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        default_for_task: false,
        backends: &[FrozenBackend {
            backend: "metal",
            engine: "owned-metal",
            family: "qwen3-0.6b",
            dtype: "f16",
            execution: "explicit",
            attention_units: FROZEN_ATTENTION_UNITS,
            max_tokens: FROZEN_MAX_TOKENS,
            pooling: Some("last"),
            normalize: Some(true),
            dims: Some(1024),
            rerank_abs_tolerance: None,
        }],
    },
    FrozenEntry {
        id: "qwen3-reranker-0.6b",
        hf_repo: "Qwen/Qwen3-Reranker-0.6B",
        revision: "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        default_for_task: false,
        backends: &[],
    },
];

/// Schema validation plus the frozen-set rules that apply only to the
/// compiled catalog of this release.
pub(crate) fn validate_release(catalog: &Catalog) -> Result<(), CatalogError> {
    validate_schema(catalog)?;
    let expected: Vec<String> = FROZEN_ENTRIES
        .iter()
        .map(|entry| entry.id.to_string())
        .collect();
    let mut actual: Vec<String> = catalog
        .models
        .iter()
        .map(|entry| entry.id.clone())
        .collect();
    actual.sort();
    if actual != expected {
        return Err(CatalogError::FrozenIdSet { expected, actual });
    }
    for frozen in &FROZEN_ENTRIES {
        let entry = catalog
            .entry(frozen.id)
            .expect("the id set was checked above");
        let mismatch = |field: String, expected: String, actual: String| {
            Err(CatalogError::FrozenMismatch {
                id: entry.id.clone(),
                field,
                expected,
                actual,
            })
        };
        if entry.upstream.hf_repo != frozen.hf_repo {
            return mismatch(
                "upstream.hf_repo".into(),
                frozen.hf_repo.into(),
                entry.upstream.hf_repo.clone(),
            );
        }
        if entry.upstream.revision != frozen.revision {
            return mismatch(
                "upstream.revision".into(),
                frozen.revision.into(),
                entry.upstream.revision.clone(),
            );
        }
        if entry.default_for_task != frozen.default_for_task {
            return mismatch(
                "default_for_task".into(),
                frozen.default_for_task.to_string(),
                entry.default_for_task.to_string(),
            );
        }
        let expected_set: Vec<&str> = frozen.backends.iter().map(|row| row.backend).collect();
        let actual_set: Vec<&str> = entry
            .backends
            .iter()
            .map(|row| row.backend.as_str())
            .collect();
        if actual_set != expected_set {
            return mismatch(
                "backends".into(),
                format!("{expected_set:?}"),
                format!("{actual_set:?}"),
            );
        }
        for (row, frozen_row) in entry.backends.iter().zip(frozen.backends) {
            let literals: [(&str, String, String); 10] = [
                ("engine", frozen_row.engine.into(), row.engine.clone()),
                (
                    "family",
                    show(Some(frozen_row.family)),
                    show(row.family.as_deref()),
                ),
                (
                    "dtype",
                    show(Some(frozen_row.dtype)),
                    show(row.dtype.as_deref()),
                ),
                (
                    "execution",
                    show(Some(frozen_row.execution)),
                    show(row.execution.as_deref()),
                ),
                (
                    "attention_units",
                    show(Some(frozen_row.attention_units)),
                    show(row.attention_units),
                ),
                (
                    "max_tokens",
                    show(Some(frozen_row.max_tokens)),
                    show(row.max_tokens),
                ),
                (
                    "pooling",
                    show(frozen_row.pooling),
                    show(row.pooling.as_deref()),
                ),
                ("normalize", show(frozen_row.normalize), show(row.normalize)),
                ("dims", show(frozen_row.dims), show(row.dims)),
                (
                    "rerank_abs_tolerance",
                    show(frozen_row.rerank_abs_tolerance),
                    show(row.rerank_abs_tolerance),
                ),
            ];
            for (field, expected, actual) in literals {
                if expected != actual {
                    return mismatch(
                        format!("backends[{}].{field}", row.backend),
                        expected,
                        actual,
                    );
                }
            }
        }
    }
    Ok(())
}

fn show<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "absent".to_string(), |value| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sha(seed: char) -> String {
        seed.to_string().repeat(64)
    }

    /// A schema-valid fixture catalog with ids outside the frozen set: one
    /// embed entry and one rerank entry that both ship a `tokenizer.json`.
    fn fixture() -> Value {
        let vector = json!([0.5, 0.5, 0.5, 0.5]);
        json!({
            "catalog_revision": 3,
            "models": [
                {
                    "id": "fixture-embed",
                    "task": "embed",
                    "name": "Fixture Embed",
                    "description": "An embedding fixture.",
                    "default_for_task": true,
                    "upstream": {"hf_repo": "fixture/embed", "revision": "a".repeat(40)},
                    "files": [
                        {"path": "model.safetensors", "role": "model", "sha256": sha('1'), "size_bytes": 100, "backends": ["metal"]},
                        {"path": "tokenizer.json", "role": "tokenizer", "sha256": sha('2'), "size_bytes": 20, "backends": ["metal"]},
                        {"path": "config.json", "role": "config", "sha256": sha('3'), "size_bytes": 3, "backends": ["metal"]}
                    ],
                    "backends": [
                        {"backend": "metal", "engine": "owned-metal", "family": "gte-modernbert", "dtype": "f16", "execution": "explicit", "attention_units": 256, "max_tokens": 16, "pooling": "cls", "normalize": true, "dims": 4, "fingerprint": sha('f')}
                    ],
                    "self_check": {
                        "fixture_revision": 1,
                        "reference_tool": "fixture tool 1.0",
                        "inputs": ["a", "b", "c", "d", "e", "f", "g", "h"],
                        "reference": {"vectors": [vector, vector, vector, vector, vector, vector, vector, vector]}
                    }
                },
                {
                    "id": "fixture-rerank",
                    "task": "rerank",
                    "name": "Fixture Rerank",
                    "description": "A reranking fixture.",
                    "default_for_task": true,
                    "upstream": {"hf_repo": "fixture/rerank", "revision": "b".repeat(40)},
                    "files": [
                        {"path": "model.safetensors", "role": "model", "sha256": sha('4'), "size_bytes": 200, "backends": ["metal"]},
                        {"path": "tokenizer.json", "role": "tokenizer", "sha256": sha('2'), "size_bytes": 20, "backends": ["metal"]}
                    ],
                    "backends": [
                        {"backend": "metal", "engine": "owned-metal", "family": "gte-modernbert", "dtype": "f32", "execution": "explicit", "attention_units": 256, "max_tokens": 16, "rerank_abs_tolerance": 0.005, "fingerprint": sha('e')}
                    ],
                    "self_check": {
                        "fixture_revision": 1,
                        "reference_tool": "fixture tool 1.0",
                        "inputs": [
                            {"query": "q1", "candidates": ["x", "y"]},
                            {"query": "q2", "candidates": ["x"]},
                            {"query": "q3", "candidates": ["x", "y", "z"]},
                            {"query": "q4", "candidates": ["x", "y"]}
                        ],
                        "reference": {"scores": [[0.9, 0.1], [0.5], [0.8, 0.2, 0.1], [0.3, 0.7]]}
                    }
                }
            ]
        })
    }

    fn schema(document: &Value) -> Result<Catalog, CatalogError> {
        load_schema_valid(&document.to_string())
    }

    /// Applies `mutate` to the valid fixture and returns the schema error.
    fn schema_error(mutate: impl FnOnce(&mut Value)) -> CatalogError {
        let mut document = fixture();
        mutate(&mut document);
        schema(&document).expect_err("mutated fixture must fail schema validation")
    }

    fn embed(document: &mut Value) -> &mut Value {
        &mut document["models"][0]
    }

    fn rerank(document: &mut Value) -> &mut Value {
        &mut document["models"][1]
    }

    fn compiled_value() -> Value {
        serde_json::from_str(COMPILED_CATALOG_JSON).unwrap()
    }

    fn compiled_entry_index(id: &str) -> usize {
        compiled_value()["models"]
            .as_array()
            .unwrap()
            .iter()
            .position(|entry| entry["id"] == id)
            .unwrap()
    }

    /// Applies `mutate` to the compiled catalog and returns the release error.
    fn release_error(mutate: impl FnOnce(&mut Value)) -> CatalogError {
        let mut document = compiled_value();
        mutate(&mut document);
        load_release_valid(&document.to_string())
            .expect_err("mutated compiled catalog must fail release validation")
    }

    #[test]
    fn fixture_catalog_is_schema_valid_with_one_path_in_two_entries() {
        let catalog = schema(&fixture()).expect("fixture is schema-valid");
        let with_tokenizer = catalog
            .models
            .iter()
            .filter(|entry| entry.files.iter().any(|file| file.path == "tokenizer.json"))
            .count();
        assert_eq!(with_tokenizer, 2);
    }

    #[test]
    fn fixture_catalog_is_not_release_valid() {
        assert!(matches!(
            load_release_valid(&fixture().to_string()),
            Err(CatalogError::FrozenIdSet { .. })
        ));
    }

    #[test]
    fn schema_rejects_duplicate_entry_ids() {
        let error = schema_error(|doc| {
            let copy = doc["models"][0].clone();
            doc["models"].as_array_mut().unwrap().push(copy);
        });
        assert_eq!(
            error,
            CatalogError::DuplicateEntryId("fixture-embed".into())
        );
    }

    #[test]
    fn schema_rejects_invalid_entry_ids() {
        let error = schema_error(|doc| embed(doc)["id"] = json!("Fixture-Embed"));
        assert!(matches!(error, CatalogError::InvalidEntryId(_)), "{error}");
    }

    #[test]
    fn schema_rejects_a_catalog_id_equal_to_another_entrys_lane_id() {
        let error = schema_error(|doc| rerank(doc)["id"] = json!("fixture-embed-cuda"));
        assert_eq!(
            error,
            CatalogError::ReservedIdCollision {
                id: "fixture-embed-cuda".into(),
                owner: "fixture-embed".into()
            }
        );
    }

    #[test]
    fn schema_rejects_two_files_with_one_path_in_a_single_entry() {
        let error = schema_error(|doc| {
            let mut copy = embed(doc)["files"][2].clone();
            copy["path"] = json!("tokenizer.json");
            embed(doc)["files"][2] = copy;
        });
        assert_eq!(
            error,
            CatalogError::DuplicateFilePath {
                id: "fixture-embed".into(),
                path: "tokenizer.json".into()
            }
        );
    }

    #[test]
    fn schema_rejects_a_duplicate_backend_in_one_entry() {
        let error = schema_error(|doc| {
            let copy = embed(doc)["backends"][0].clone();
            embed(doc)["backends"].as_array_mut().unwrap().push(copy);
        });
        assert_eq!(
            error,
            CatalogError::DuplicateBackend {
                id: "fixture-embed".into(),
                backend: "metal".into()
            }
        );
    }

    #[test]
    fn schema_rejects_backend_values_outside_the_allowed_set() {
        for value in ["cpu", "llama", "ort", "Metal"] {
            let error = schema_error(|doc| embed(doc)["backends"][0]["backend"] = json!(value));
            assert!(
                matches!(&error, CatalogError::UnknownBackend { backend, .. } if backend == value),
                "{value}: {error}"
            );
        }
    }

    #[test]
    fn schema_rejects_engines_outside_the_allowed_set() {
        for value in ["ort", "llama", "cpu"] {
            let error = schema_error(|doc| embed(doc)["backends"][0]["engine"] = json!(value));
            assert!(
                matches!(&error, CatalogError::UnknownEngine { engine, .. } if engine == value),
                "{value}: {error}"
            );
        }
    }

    #[test]
    fn schema_rejects_unknown_roles() {
        let error = schema_error(|doc| embed(doc)["files"][2]["role"] = json!("weights"));
        assert!(matches!(error, CatalogError::UnknownRole { .. }), "{error}");
    }

    #[test]
    fn schema_rejects_role_cardinality_violations() {
        let two_models = schema_error(|doc| embed(doc)["files"][2]["role"] = json!("model"));
        assert!(
            matches!(
                &two_models,
                CatalogError::RoleCardinality {
                    role: "model",
                    count: 2,
                    ..
                }
            ),
            "{two_models}"
        );
        let no_tokenizer = schema_error(|doc| {
            embed(doc)["files"].as_array_mut().unwrap().remove(1);
        });
        assert!(
            matches!(
                &no_tokenizer,
                CatalogError::RoleCardinality {
                    role: "tokenizer",
                    count: 0,
                    ..
                }
            ),
            "{no_tokenizer}"
        );
        let two_configs = schema_error(|doc| {
            let mut extra = embed(doc)["files"][2].clone();
            extra["path"] = json!("config-2.json");
            embed(doc)["files"].as_array_mut().unwrap().push(extra);
        });
        assert!(
            matches!(
                &two_configs,
                CatalogError::RoleCardinality {
                    role: "config",
                    count: 2,
                    ..
                }
            ),
            "{two_configs}"
        );
    }

    #[test]
    fn schema_rejects_empty_or_undeclared_file_backends() {
        let empty = schema_error(|doc| embed(doc)["files"][0]["backends"] = json!([]));
        assert!(
            matches!(empty, CatalogError::FileBackendsEmpty { .. }),
            "{empty}"
        );
        let undeclared =
            schema_error(|doc| embed(doc)["files"][0]["backends"] = json!(["metal", "ane"]));
        assert!(
            matches!(&undeclared, CatalogError::FileBackendUndeclared { backend, .. } if backend == "ane"),
            "{undeclared}"
        );
    }

    #[test]
    fn schema_rejects_embed_only_fields_on_rerank_backends() {
        for (field, value) in [
            ("pooling", json!("cls")),
            ("normalize", json!(true)),
            ("dims", json!(768)),
        ] {
            let error = schema_error(|doc| rerank(doc)["backends"][0][field] = value);
            assert!(
                matches!(&error, CatalogError::EmbedFieldOnRerank { field: found, .. } if *found == field),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn schema_rejects_embed_backends_missing_embed_fields() {
        for field in ["pooling", "normalize", "dims"] {
            let error = schema_error(|doc| {
                embed(doc)["backends"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove(field);
            });
            assert!(
                matches!(&error, CatalogError::MissingBackendField { field: found, .. } if *found == field),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn schema_rejects_rerank_tolerance_on_embed_backends() {
        let error =
            schema_error(|doc| embed(doc)["backends"][0]["rerank_abs_tolerance"] = json!(0.01));
        assert!(
            matches!(error, CatalogError::RerankFieldOnEmbed { .. }),
            "{error}"
        );
    }

    #[test]
    fn schema_rejects_rerank_backends_without_a_finite_positive_tolerance() {
        let missing = schema_error(|doc| {
            rerank(doc)["backends"][0]
                .as_object_mut()
                .unwrap()
                .remove("rerank_abs_tolerance");
        });
        assert!(
            matches!(
                &missing,
                CatalogError::MissingBackendField {
                    field: "rerank_abs_tolerance",
                    ..
                }
            ),
            "{missing}"
        );
        for value in [0.0, -0.005] {
            let error = schema_error(|doc| {
                rerank(doc)["backends"][0]["rerank_abs_tolerance"] = json!(value)
            });
            assert!(
                matches!(
                    &error,
                    CatalogError::InvalidBackendField {
                        field: "rerank_abs_tolerance",
                        ..
                    }
                ),
                "{value}: {error}"
            );
        }
    }

    #[test]
    fn schema_rejects_backends_missing_load_fields() {
        for field in [
            "family",
            "dtype",
            "execution",
            "attention_units",
            "max_tokens",
        ] {
            let error = schema_error(|doc| {
                embed(doc)["backends"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove(field);
            });
            assert!(
                matches!(&error, CatalogError::MissingBackendField { field: found, .. } if *found == field),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn schema_rejects_attention_units_below_max_tokens_squared() {
        let error = schema_error(|doc| embed(doc)["backends"][0]["attention_units"] = json!(255));
        assert_eq!(
            error,
            CatalogError::AttentionUnitsBelowContext {
                id: "fixture-embed".into(),
                backend: "metal".into(),
                attention_units: 255,
                max_tokens: 16
            }
        );
    }

    #[test]
    fn schema_rejects_revisions_that_are_not_40_lowercase_hex() {
        for revision in [
            "main".to_string(),
            "a".repeat(39),
            "A".repeat(40),
            "g".repeat(40),
        ] {
            let error = schema_error(|doc| embed(doc)["upstream"]["revision"] = json!(revision));
            assert!(
                matches!(error, CatalogError::InvalidRevision { .. }),
                "{revision}: {error}"
            );
        }
    }

    #[test]
    fn schema_rejects_sha256_and_fingerprints_that_are_not_64_lowercase_hex() {
        for bad in ["A".repeat(64), "a".repeat(63), "z".repeat(64)] {
            let sha = schema_error(|doc| embed(doc)["files"][0]["sha256"] = json!(bad));
            assert!(
                matches!(sha, CatalogError::InvalidSha256 { .. }),
                "{bad}: {sha}"
            );
            let fingerprint =
                schema_error(|doc| embed(doc)["backends"][0]["fingerprint"] = json!(bad));
            assert!(
                matches!(fingerprint, CatalogError::InvalidFingerprint { .. }),
                "{bad}: {fingerprint}"
            );
        }
    }

    #[test]
    fn schema_rejects_self_check_against_the_backends_rule() {
        let missing = schema_error(|doc| {
            embed(doc).as_object_mut().unwrap().remove("self_check");
        });
        assert!(
            matches!(missing, CatalogError::SelfCheckMissing { .. }),
            "{missing}"
        );
        let unexpected = schema_error(|doc| {
            let entry = rerank(doc);
            entry["backends"] = json!([]);
            entry["files"] = json!([]);
        });
        assert!(
            matches!(unexpected, CatalogError::SelfCheckUnexpected { .. }),
            "{unexpected}"
        );
    }

    #[test]
    fn schema_accepts_an_entry_with_no_backends_files_or_self_check() {
        let mut document = fixture();
        let entry = rerank(&mut document);
        entry["backends"] = json!([]);
        entry["files"] = json!([]);
        entry.as_object_mut().unwrap().remove("self_check");
        schema(&document).expect("a listed, never-runnable entry is schema-valid");
    }

    fn self_check_reason(error: CatalogError) -> String {
        match error {
            CatalogError::SelfCheckShape { reason, .. } => reason,
            other => panic!("expected a self_check shape error, got {other}"),
        }
    }

    #[test]
    fn schema_rejects_embed_self_checks_without_8_inputs_of_dims_vectors() {
        let seven = schema_error(|doc| {
            let check = &mut embed(doc)["self_check"];
            check["inputs"].as_array_mut().unwrap().pop();
            check["reference"]["vectors"].as_array_mut().unwrap().pop();
        });
        assert!(self_check_reason(seven).contains("needs 8 inputs"));
        let short_vector = schema_error(|doc| {
            embed(doc)["self_check"]["reference"]["vectors"][5] = json!([0.5, 0.5, 0.5]);
        });
        assert!(self_check_reason(short_vector).contains("reference vector 5 has 3 values"));
        let seven_vectors = schema_error(|doc| {
            embed(doc)["self_check"]["reference"]["vectors"]
                .as_array_mut()
                .unwrap()
                .pop();
        });
        assert!(self_check_reason(seven_vectors).contains("7 reference vectors"));
    }

    #[test]
    fn schema_rejects_rerank_self_checks_without_4_inputs_and_matching_scores() {
        let three = schema_error(|doc| {
            let check = &mut rerank(doc)["self_check"];
            check["inputs"].as_array_mut().unwrap().pop();
            check["reference"]["scores"].as_array_mut().unwrap().pop();
        });
        assert!(self_check_reason(three).contains("needs 4 inputs"));
        let short_row = schema_error(|doc| {
            rerank(doc)["self_check"]["reference"]["scores"][2] = json!([0.8, 0.2]);
        });
        assert!(self_check_reason(short_row).contains("input 2 has 3 candidates but 2 scores"));
    }

    #[test]
    fn schema_rejects_non_positive_catalog_revisions() {
        for revision in [0, -1] {
            let error = schema_error(|doc| doc["catalog_revision"] = json!(revision));
            assert_eq!(error, CatalogError::NonPositiveCatalogRevision(revision));
        }
    }

    #[test]
    fn schema_requires_exactly_one_default_per_task() {
        let none = schema_error(|doc| rerank(doc)["default_for_task"] = json!(false));
        assert_eq!(
            none,
            CatalogError::DefaultForTask {
                task: "rerank".into(),
                count: 0
            }
        );
        let two = schema_error(|doc| {
            let mut copy = doc["models"][0].clone();
            copy["id"] = json!("fixture-embed-two");
            doc["models"].as_array_mut().unwrap().push(copy);
        });
        assert_eq!(
            two,
            CatalogError::DefaultForTask {
                task: "embed".into(),
                count: 2
            }
        );
    }

    #[test]
    fn schema_rejects_embed_entries_that_mention_rerank() {
        for field in ["id", "name", "description"] {
            let error = schema_error(|doc| embed(doc)[field] = json!("fixture-reranker"));
            assert!(
                matches!(error, CatalogError::RerankWordInEmbedEntry { .. }),
                "{field}: {error}"
            );
        }
        let upper = schema_error(|doc| embed(doc)["name"] = json!("Fixture Reranker"));
        assert!(
            matches!(upper, CatalogError::RerankWordInEmbedEntry { .. }),
            "{upper}"
        );
    }

    #[test]
    fn schema_rejects_unknown_fields() {
        let error = schema_error(|doc| embed(doc)["backends"][0]["quant"] = json!("q8"));
        assert!(matches!(error, CatalogError::Malformed(_)), "{error}");
    }

    #[test]
    fn compiled_catalog_passes_release_validation() {
        let catalog = compiled_catalog().expect("compiled models.json is release-valid");
        assert_eq!(catalog.catalog_revision, 1);
    }

    #[test]
    fn compiled_catalog_declares_the_four_frozen_entries() {
        let catalog = compiled_catalog().unwrap();
        let ids: Vec<&str> = catalog
            .models
            .iter()
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(
            ids,
            [
                "gte-modernbert-base",
                "gte-reranker-modernbert-base",
                "qwen3-embedding-0.6b",
                "qwen3-reranker-0.6b"
            ]
        );
        for id in [
            "gte-modernbert-base",
            "gte-reranker-modernbert-base",
            "qwen3-embedding-0.6b",
        ] {
            let entry = catalog.entry(id).unwrap();
            assert_eq!(entry.backends.len(), 1, "{id}");
            let metal = entry.backend("metal").unwrap();
            assert_eq!(metal.engine, "owned-metal");
            assert_eq!(metal.attention_units, Some(67_108_864));
            assert_eq!(metal.max_tokens, Some(8192));
            assert_ne!(
                metal.fingerprint,
                "0".repeat(64),
                "{id} fingerprint was minted"
            );
            let files = entry.backend_files("metal");
            assert_eq!(
                files.keys().copied().collect::<Vec<_>>(),
                ["config", "model", "tokenizer"],
                "{id}"
            );
            assert!(files.values().all(|file| file.size_bytes > 0));
            let check = entry.self_check.as_ref().unwrap();
            assert_eq!(check.fixture_revision, 1);
        }
        let qwen_reranker = catalog.entry("qwen3-reranker-0.6b").unwrap();
        assert!(qwen_reranker.backends.is_empty());
        assert!(qwen_reranker.files.is_empty());
        assert!(qwen_reranker.self_check.is_none());
        assert_eq!(
            qwen_reranker.upstream.revision,
            "e61197ed45024b0ed8a2d74b80b4d909f1255473"
        );
        assert_eq!(
            catalog
                .entry("gte-reranker-modernbert-base")
                .unwrap()
                .upstream
                .revision,
            "f7481e6055501a30fb19d090657df9ec1f79ab2c"
        );
    }

    #[test]
    fn release_rejects_id_sets_other_than_the_frozen_four() {
        let missing = release_error(|doc| {
            doc["models"].as_array_mut().unwrap().pop();
        });
        assert!(
            matches!(missing, CatalogError::FrozenIdSet { .. }),
            "{missing}"
        );
        let extra = release_error(|doc| {
            let mut copy = doc["models"][3].clone();
            copy["id"] = json!("minilm");
            doc["models"].as_array_mut().unwrap().push(copy);
        });
        assert!(matches!(extra, CatalogError::FrozenIdSet { .. }), "{extra}");
    }

    #[test]
    fn release_rejects_the_gte_reranker_revision_on_qwen3_reranker() {
        let index = compiled_entry_index("qwen3-reranker-0.6b");
        let error = release_error(|doc| {
            doc["models"][index]["upstream"]["revision"] =
                json!("f7481e6055501a30fb19d090657df9ec1f79ab2c");
        });
        assert!(
            matches!(&error, CatalogError::FrozenMismatch { id, field, .. }
                if id == "qwen3-reranker-0.6b" && field == "upstream.revision"),
            "{error}"
        );
    }

    #[test]
    fn release_rejects_a_changed_repo_or_revision() {
        let index = compiled_entry_index("gte-modernbert-base");
        let revision = release_error(|doc| {
            doc["models"][index]["upstream"]["revision"] = json!("0".repeat(40));
        });
        assert!(
            matches!(&revision, CatalogError::FrozenMismatch { field, .. } if field == "upstream.revision"),
            "{revision}"
        );
        let repo = release_error(|doc| {
            doc["models"][index]["upstream"]["hf_repo"] = json!("Alibaba-NLP/gte-modernbert-large");
        });
        assert!(
            matches!(&repo, CatalogError::FrozenMismatch { field, .. } if field == "upstream.hf_repo"),
            "{repo}"
        );
    }

    #[test]
    fn release_rejects_backend_sets_outside_the_frozen_set() {
        let index = compiled_entry_index("gte-modernbert-base");
        let cuda = release_error(|doc| {
            let entry = &mut doc["models"][index];
            let mut row = entry["backends"][0].clone();
            row["backend"] = json!("cuda");
            entry["backends"].as_array_mut().unwrap().push(row);
            for file in entry["files"].as_array_mut().unwrap() {
                file["backends"] = json!(["metal", "cuda"]);
            }
        });
        assert!(
            matches!(&cuda, CatalogError::FrozenMismatch { field, .. } if field == "backends"),
            "{cuda}"
        );
        let qwen_reranker = compiled_entry_index("qwen3-reranker-0.6b");
        let gte_reranker = compiled_entry_index("gte-reranker-modernbert-base");
        let runnable_qwen_reranker = release_error(|doc| {
            let donor = doc["models"][gte_reranker].clone();
            let entry = &mut doc["models"][qwen_reranker];
            entry["backends"] = donor["backends"].clone();
            entry["files"] = donor["files"].clone();
            entry["self_check"] = donor["self_check"].clone();
        });
        assert!(
            matches!(&runnable_qwen_reranker, CatalogError::FrozenMismatch { id, field, .. }
                if id == "qwen3-reranker-0.6b" && field == "backends"),
            "{runnable_qwen_reranker}"
        );
    }

    #[test]
    fn release_rejects_load_literals_outside_the_frozen_set() {
        let cases: [(&str, &str, Value); 7] = [
            ("gte-modernbert-base", "dtype", json!("f32")),
            ("gte-modernbert-base", "pooling", json!("mean")),
            ("gte-modernbert-base", "execution", json!("lazy")),
            (
                "gte-reranker-modernbert-base",
                "rerank_abs_tolerance",
                json!(0.01),
            ),
            ("gte-reranker-modernbert-base", "max_tokens", json!(4096)),
            ("qwen3-embedding-0.6b", "family", json!("gte-modernbert")),
            ("qwen3-embedding-0.6b", "dims", json!(768)),
        ];
        for (id, field, value) in cases {
            let index = compiled_entry_index(id);
            let mut document = compiled_value();
            document["models"][index]["backends"][0][field] = value;
            if field == "dims" {
                for vector in document["models"][index]["self_check"]["reference"]["vectors"]
                    .as_array_mut()
                    .unwrap()
                {
                    vector.as_array_mut().unwrap().truncate(768);
                }
            }
            let error = load_release_valid(&document.to_string()).unwrap_err();
            let expected_field = format!("backends[metal].{field}");
            assert!(
                matches!(&error, CatalogError::FrozenMismatch { id: found, field, .. }
                    if found == id && *field == expected_field),
                "{id}.{field}: {error}"
            );
        }
    }

    #[test]
    fn release_rejects_changed_defaults() {
        let index = compiled_entry_index("gte-reranker-modernbert-base");
        let qwen = compiled_entry_index("qwen3-reranker-0.6b");
        let default = release_error(|doc| {
            doc["models"][index]["default_for_task"] = json!(false);
            doc["models"][qwen]["default_for_task"] = json!(true);
        });
        assert!(
            matches!(&default, CatalogError::FrozenMismatch { field, .. } if field == "default_for_task"),
            "{default}"
        );
    }

    #[test]
    fn manifest_digest_is_the_sha256_of_the_jcs_manifest_and_is_stable() {
        // Expected values were computed independently in Python with
        // json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        // over {upstream, files}, which equals JCS for these ASCII-keyed,
        // integer-only documents.
        let catalog = compiled_catalog().unwrap();
        let expected = [
            (
                "gte-modernbert-base",
                "6c0959a4db05e22d96838f326cc98d70dc5caa7a4c70f0a7ff3eba2538319672",
            ),
            (
                "gte-reranker-modernbert-base",
                "f0f4c8f6497d270dd11216be43606e2de0b35935f062c59d842ad220a311720b",
            ),
            (
                "qwen3-embedding-0.6b",
                "3209ee0c53b1b2ab42602e31c43b0c369a71d5dbb8343e73b8000a30cd533994",
            ),
            (
                "qwen3-reranker-0.6b",
                "8132740ff0ac2b86c9d811661c0025db4cf3dc20283672314ffa6dfa673d863c",
            ),
        ];
        for (id, digest) in expected {
            assert_eq!(catalog.entry(id).unwrap().manifest_digest(), digest, "{id}");
        }
    }

    #[test]
    fn manifest_digest_ignores_document_key_order_and_tracks_file_content() {
        let catalog = schema(&fixture()).unwrap();
        let entry = &catalog.models[0];
        let mut reordered = serde_json::Map::new();
        let original = serde_json::to_value(entry).unwrap();
        for (key, value) in original.as_object().unwrap().iter().rev() {
            reordered.insert(key.clone(), value.clone());
        }
        let reparsed: CatalogEntry = serde_json::from_value(Value::Object(reordered)).unwrap();
        assert_eq!(reparsed.manifest_digest(), entry.manifest_digest());

        let mut changed = entry.clone();
        changed.files[0].sha256 = sha('9');
        assert_ne!(changed.manifest_digest(), entry.manifest_digest());
        let mut fingerprint_only = entry.clone();
        fingerprint_only.backends[0].fingerprint = sha('0');
        assert_eq!(
            fingerprint_only.manifest_digest(),
            entry.manifest_digest(),
            "backends are not part of the manifest"
        );
    }

    #[test]
    fn jcs_sorts_by_utf16_code_units_and_escapes_minimally() {
        // U+1F600 encodes as the surrogate 0xD83D, which sorts before U+E000
        // in UTF-16 even though its code point is larger.
        let value =
            json!({"\u{e000}": 1, "\u{1f600}": 2, "b": [true, null, -3], "a": "\u{7}\n\"/é"});
        assert_eq!(
            jcs(&value).unwrap(),
            "{\"a\":\"\\u0007\\n\\\"/é\",\"b\":[true,null,-3],\"\u{1f600}\":2,\"\u{e000}\":1}"
        );
    }

    #[test]
    fn jcs_refuses_numbers_it_cannot_serialize_canonically() {
        assert!(jcs(&json!(0.5)).is_err());
        assert!(jcs(&json!(1u64 << 53)).is_err());
        assert_eq!(jcs(&json!((1u64 << 53) - 1)).unwrap(), "9007199254740991");
    }

    #[test]
    fn reserved_ids_cover_catalog_ids_and_every_allowed_backend_suffix() {
        let catalog = compiled_catalog().unwrap();
        let (entry, backend) = catalog.resolve_reserved("gte-modernbert-base").unwrap();
        assert_eq!((entry.id.as_str(), backend), ("gte-modernbert-base", None));
        let (entry, backend) = catalog.resolve_reserved("gte-modernbert-base-ane").unwrap();
        assert_eq!(
            (entry.id.as_str(), backend),
            ("gte-modernbert-base", Some("ane"))
        );
        assert!(catalog.is_reserved_id("qwen3-reranker-0.6b-metal"));
        assert!(!catalog.is_reserved_id("gte-modernbert-base-f16"));
        assert!(!catalog.is_reserved_id("gte-modernbert-base-cpu"));
        assert!(!catalog.is_reserved_id("minilm"));
        assert_eq!(
            lane_id("qwen3-embedding-0.6b", "metal"),
            "qwen3-embedding-0.6b-metal"
        );
    }

    #[test]
    fn download_bytes_sums_distinct_files_of_runnable_backends() {
        let catalog = compiled_catalog().unwrap();
        let entry = catalog.entry("gte-modernbert-base").unwrap();
        assert_eq!(
            entry.download_bytes(&["metal"]),
            298_041_568 + 3_583_228 + 1_184
        );
        assert_eq!(entry.download_bytes(&[]), 0);
        assert_eq!(entry.download_bytes(&["cuda"]), 0);
        let unservable = catalog.entry("qwen3-reranker-0.6b").unwrap();
        assert_eq!(unservable.download_bytes(&["metal"]), 0);
    }
}
