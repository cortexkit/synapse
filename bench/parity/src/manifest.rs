//! Typed form of `bench/parity/models.json`.
//!
//! Every struct refuses unknown fields, so a misspelled or stray key fails
//! schema validation instead of being silently ignored. Optional profile keys
//! are omitted (not `null`) when absent, because absence is meaningful: a
//! profile digest must differ between "no rotation" and `rotation: "none"`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::canonical::{canonical_bytes, canonical_bytes_of, sha256_hex};
use crate::{perr, Result};

pub const SCHEMA: &str = "synapse-parity-models-v1";
pub const MANIFEST_FILE: &str = "models.json";

/// The four model slugs this manifest covers. Order does not matter; every
/// check compares them as a set.
pub const MODEL_SLUGS: [&str; 4] = [
    "gte-modernbert-base",
    "gte-reranker-modernbert-base",
    "qwen3-embedding-0.6b",
    "qwen3-reranker-0.6b",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub reference: Reference,
    pub admission: Admission,
    pub converter: Converter,
    pub rotations: BTreeMap<String, Rotation>,
    pub models: BTreeMap<String, Model>,
    pub profiles: BTreeMap<String, Profile>,
    pub digests: Digests,
}

/// The CPU fp32 reference every parity fixture is generated from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub reference_transformers_version: String,
    pub reference_seed: u64,
    pub device: String,
    pub dtype: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Admission {
    pub max_context_tokens: u32,
    pub ane_resident_shapes_per_model: u32,
    pub ane_resident_shapes_total: u32,
    pub ane_shape_ladder: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Converter {
    pub rule: String,
    pub command: String,
}

/// A rotation matrix generated in code (`hadamard::generate`); `sha256` pins
/// its row-major little-endian f64 bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rotation {
    pub sha256: String,
    pub dimension: u32,
    pub element: String,
    pub layout: String,
    pub sign_seed: u64,
    pub construction: String,
    pub scope: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Embed,
    Rerank,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Modernbert,
    Qwen3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    /// First position (the CLS token).
    Cls,
    /// Attention-masked mean over non-pad positions.
    MaskedMean,
    /// Last non-pad position.
    LastNonPad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalization {
    L2,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Lane {
    #[serde(rename = "owned-metal")]
    OwnedMetal,
    #[serde(rename = "owned-cuda")]
    OwnedCuda,
    #[serde(rename = "owned-vulkan")]
    OwnedVulkan,
    #[serde(rename = "ane-direct-worker")]
    AneDirect,
}

impl Lane {
    pub const ALL: [Lane; 4] = [
        Lane::OwnedMetal,
        Lane::OwnedCuda,
        Lane::OwnedVulkan,
        Lane::AneDirect,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Lane::OwnedMetal => "owned-metal",
            Lane::OwnedCuda => "owned-cuda",
            Lane::OwnedVulkan => "owned-vulkan",
            Lane::AneDirect => "ane-direct-worker",
        }
    }

    /// Worker lanes load a converted package; in-process Metal loads the
    /// pinned checkpoint and has no package.
    pub fn is_worker(self) -> bool {
        !matches!(self, Lane::OwnedMetal)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToleranceClass {
    Fp32,
    Fp16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub hf_repo: String,
    pub hf_revision: String,
    pub operation: Operation,
    /// Prefix the checkpoint puts before its backbone tensor names (`model.`
    /// on the two rerankers, empty on the two embedders).
    pub tensor_prefix: String,
    /// SHA-256 (lowercase hex) of each pinned checkpoint file, by file name.
    pub files: BTreeMap<String, String>,
    /// SHA-256 of the raw JSON header inside `model.safetensors`; a committed
    /// copy lives at `checkpoints/<slug>/tensor-index.json`.
    pub tensor_index_sha256: String,
    pub checkpoint_digest: String,
    pub tokenizer_digest: String,
    pub architecture: Architecture,
    pub grammar: Grammar,
    pub output: Output,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<Head>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Architecture {
    /// The checkpoint's `architectures[0]` entry.
    pub class: String,
    pub family: Family,
    /// Parameter name to value. The allowed names, their JSON types and the
    /// checkpoint field each comes from are fixed per family in `arch`.
    pub params: BTreeMap<String, Value>,
}

impl Architecture {
    pub fn int(&self, name: &str) -> Result<u64> {
        self.params
            .get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| perr!("architecture parameter `{name}` is missing or not an integer"))
    }

    pub fn boolean(&self, name: &str) -> Result<bool> {
        self.params
            .get(name)
            .and_then(Value::as_bool)
            .ok_or_else(|| perr!("architecture parameter `{name}` is missing or not a boolean"))
    }

    pub fn float(&self, name: &str) -> Result<f64> {
        self.params
            .get(name)
            .and_then(Value::as_f64)
            .ok_or_else(|| perr!("architecture parameter `{name}` is missing or not a number"))
    }
}

/// A token pinned by its text and its id in the pinned tokenizer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenRef {
    pub text: String,
    pub id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarKind {
    /// The tokenizer's single-sequence encoding with its special tokens.
    SingleSequence,
    /// The tokenizer's pair encoding with its special tokens.
    Pair,
    /// Literal template segments, each tokenized on its own without special
    /// tokens, concatenated in order.
    Template,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadoutKind {
    /// Pooled final hidden state, normalized per `output.normalization`.
    PooledHiddenState,
    /// `sigmoid(classifier logit)`.
    SigmoidClassifierLogit,
    /// `exp(l_yes) / (exp(l_yes) + exp(l_no))` at the pooled position.
    YesNoTwoWaySoftmax,
}

/// Everything that decides the token ids a model receives and how its output
/// is read. Its canonical SHA-256 is the `input_grammar` fingerprint input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grammar {
    pub kind: GrammarKind,
    /// Special tokens the encoding places, by role.
    pub special_tokens: BTreeMap<String, TokenRef>,
    /// Tokens every final input ends with, in order (empty for the Qwen3
    /// reranker, whose template ends in literal text).
    pub terminal_tokens: Vec<TokenRef>,
    pub pad: TokenRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<Template>,
    pub pooling: Pooling,
    pub readout: Readout,
}

/// Literal template strings, pinned against a committed copy of the upstream
/// model card.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Template {
    pub prefix: String,
    pub instruction: String,
    /// Body with `{instruction}`, `{query}` and `{doc}` placeholders.
    pub body_format: String,
    pub suffix: String,
    pub oracle: Oracle,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Oracle {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Readout {
    pub kind: ReadoutKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<TokenRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no: Option<TokenRef>,
    /// Tensor whose `yes`/`no` rows are the readout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Output {
    pub dimension: u32,
    pub normalization: Normalization,
}

/// A tensor the head reads, with its shape in the checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadTensor {
    pub key: String,
    pub shape: Vec<u64>,
}

/// Rerank head. ModernBERT: dense, activation, norm, classifier. Qwen3: the
/// `yes`/`no` rows of the readout matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Head {
    pub pooling: Pooling,
    /// Head stages in execution order, e.g. `dense`, `activation:gelu`,
    /// `layernorm`, `classifier`, or `readout_rows`.
    pub stages: Vec<String>,
    pub tensors: BTreeMap<String, HeadTensor>,
    /// Every tensor key the head needs present in the checkpoint.
    pub required_tensor_keys: Vec<String>,
    /// Keys that must be absent (biases the config switches off, or an
    /// `lm_head.weight` that would make a tied readout ambiguous).
    pub forbidden_tensor_keys: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_eps: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Fp32Tensors {
    /// Every tensor is stored in fp32.
    All(AllMarker),
    /// Exactly these tensors (package names) are kept in fp32.
    List(Vec<String>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllMarker {
    All,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub model: String,
    pub lane: Lane,
    pub storage_dtype: DType,
    pub compute_dtype: DType,
    pub fp32_tensors: Fp32Tensors,
    pub rerank_tolerance_class: ToleranceClass,
    /// `v1` for worker profiles (see `convert`); `none` for in-process Metal,
    /// which loads the pinned checkpoint itself.
    pub conversion_rule: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub converted_package_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_stages: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gelu_lowering: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan_sub_batch_max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan_min_storage_buffer_range: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan_min_device_local_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_min_driver_api: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_min_compute_major: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_min_compute_minor: Option<u32>,
}

/// Digests recorded in the manifest. They are outside every hashed entry, so
/// recording them never changes what they hash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Digests {
    pub grammar: BTreeMap<String, String>,
    pub profiles: BTreeMap<String, String>,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).map_err(|error| perr!("read {}: {error}", path.display()))?;
        Self::from_slice(&bytes)
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|error| perr!("models.json schema: {error}"))
    }

    /// The manifest file as committed: two-space pretty JSON with sorted keys
    /// and a trailing newline. `validate` requires the file to be exactly
    /// these bytes, so hand edits that reorder or reformat are caught.
    pub fn to_pretty_bytes(&self) -> Vec<u8> {
        let canonical: Value =
            serde_json::from_slice(&canonical_bytes_of(self)).expect("canonical bytes parse back");
        let mut bytes =
            serde_json::to_vec_pretty(&sorted(&canonical)).expect("manifest serializes");
        bytes.push(b'\n');
        bytes
    }

    /// SHA-256 of the whole manifest's canonical bytes: the digest workers
    /// embed and send in HELLO.
    pub fn manifest_digest(&self) -> String {
        sha256_hex(&canonical_bytes_of(self))
    }

    pub fn model(&self, slug: &str) -> Result<&Model> {
        self.models
            .get(slug)
            .ok_or_else(|| perr!("models.json has no model `{slug}`"))
    }

    /// Canonical grammar entry for one model; its SHA-256 is the grammar digest.
    pub fn grammar_entry(&self, slug: &str) -> Result<Value> {
        let model = self.model(slug)?;
        Ok(json!({"model": slug, "grammar": model.grammar}))
    }

    pub fn grammar_digest(&self, slug: &str) -> Result<String> {
        Ok(sha256_hex(&canonical_bytes(&self.grammar_entry(slug)?)))
    }

    /// Canonical manifest entry for one profile: the whole model entry
    /// (revision, digests, architecture parameters, grammar with template
    /// literals and yes/no ids, head), every per-profile key, the rotation the
    /// profile names, the admission constants and the converter rule. Its
    /// SHA-256 is the `manifest_profile_digest` fingerprint input.
    pub fn profile_entry(&self, profile_id: &str) -> Result<Value> {
        let profile = self
            .profiles
            .get(profile_id)
            .ok_or_else(|| perr!("models.json has no profile `{profile_id}`"))?;
        let model = self.model(&profile.model)?;
        let rotation =
            match profile.rotation.as_deref() {
                None | Some("none") => Value::Null,
                Some(name) => serde_json::to_value(self.rotations.get(name).ok_or_else(|| {
                    perr!("profile `{profile_id}` names unknown rotation `{name}`")
                })?)
                .expect("rotation serializes"),
            };
        Ok(json!({
            "schema": self.schema,
            "profile_id": profile_id,
            "profile": profile,
            "model_slug": profile.model,
            "model": model,
            "rotation": rotation,
            "admission": self.admission,
            "converter_rule": self.converter.rule,
        }))
    }

    pub fn profile_digest(&self, profile_id: &str) -> Result<String> {
        Ok(sha256_hex(&canonical_bytes(
            &self.profile_entry(profile_id)?,
        )))
    }

    /// Recompute every grammar and profile digest.
    pub fn computed_digests(&self) -> Result<Digests> {
        let mut grammar = BTreeMap::new();
        for slug in self.models.keys() {
            grammar.insert(slug.clone(), self.grammar_digest(slug)?);
        }
        let mut profiles = BTreeMap::new();
        for id in self.profiles.keys() {
            profiles.insert(id.clone(), self.profile_digest(id)?);
        }
        Ok(Digests { grammar, profiles })
    }
}

/// Profile id for a model on a lane: `<slug>.<lane identity>`.
pub fn profile_id(slug: &str, lane: Lane) -> String {
    format!("{slug}.{}", lane.as_str())
}

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let mut out = serde_json::Map::new();
            for (key, item) in entries {
                out.insert(key.clone(), sorted(item));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}
