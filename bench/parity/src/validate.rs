//! Schema-only validation of `models.json`.
//!
//! Needs no download: besides the manifest it reads only files committed
//! under `bench/parity/` (the copies of each pinned checkpoint's
//! `config.json`, `tokenizer_config.json` and tensor index, whose digests the
//! manifest pins, the template oracle and the rotation matrix). Checks that
//! need the full checkpoint, such as tokenizer behaviour and the weights
//! themselves, live in `checkpoint`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::Value;

use crate::arch::{check_against_config, check_schema, expected_tensors};
use crate::canonical::{is_sha256_hex, sha256_hex};
use crate::hadamard;
use crate::manifest::{
    profile_id, Family, Grammar, GrammarKind, Head, Lane, Manifest, Model, Normalization,
    Operation, Pooling, ReadoutKind, MANIFEST_FILE, MODEL_SLUGS, SCHEMA,
};
use crate::oracle::check_template;
use crate::rules::{check_profile, MODERNBERT_ROTATION};
use crate::safetensors::parse_header_json;
use crate::vulkan::{vulkan_floors, FLOOR_SEQUENCES};
use crate::{perr, Result};

/// Values the manifest must carry exactly. They are part of every profile
/// digest, so changing one is a deliberate re-pin, never a drift.
pub const REFERENCE_TRANSFORMERS_VERSION: &str = "5.16.1";
pub const REFERENCE_SEED: u64 = 0;
pub const MAX_CONTEXT_TOKENS: u32 = 8192;
pub const ANE_RESIDENT_SHAPES_PER_MODEL: u32 = 4;
pub const ANE_RESIDENT_SHAPES_TOTAL: u32 = 8;
pub const ANE_SHAPE_LADDER: [u32; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];
pub const MODERNBERT_ROTATION_SHA256: &str =
    "5cd34f0d01ce33615f987bd62be88b8e595f5083a3c94b42291c053d5cf87d53";
pub const VULKAN_SUB_BATCH_MAX_TOKENS: u32 = 8192;

/// Load `models.json` from `parity_dir` and run every schema-only check,
/// including that the file is byte-for-byte its own canonical pretty form.
pub fn validate_dir(parity_dir: &Path) -> Result<Manifest> {
    let path = parity_dir.join(MANIFEST_FILE);
    let bytes = std::fs::read(&path).map_err(|error| perr!("read {}: {error}", path.display()))?;
    let manifest = Manifest::from_slice(&bytes)?;
    validate_manifest(&manifest, parity_dir)?;
    if manifest.to_pretty_bytes() != bytes {
        return Err(perr!(
            "{} is not in canonical form; rewrite it with `parity-manifest generate`",
            path.display()
        ));
    }
    Ok(manifest)
}

pub fn validate_manifest(manifest: &Manifest, parity_dir: &Path) -> Result<()> {
    if manifest.schema != SCHEMA {
        return Err(perr!("schema `{}` is not `{SCHEMA}`", manifest.schema));
    }
    check_reference_and_admission(manifest)?;
    check_rotations(manifest)?;

    let slugs: BTreeSet<&str> = manifest.models.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = MODEL_SLUGS.into_iter().collect();
    if slugs != expected {
        return Err(perr!(
            "models must be exactly {expected:?}, found {slugs:?}"
        ));
    }
    for (slug, model) in &manifest.models {
        check_model(slug, model, parity_dir).map_err(|error| perr!("model `{slug}`: {error}"))?;
    }
    check_profiles(manifest)?;

    let computed = manifest.computed_digests()?;
    if computed != manifest.digests {
        return Err(perr!(
            "recorded digests differ from the recomputed ones; rewrite with `parity-manifest generate`\n  recorded: {:?}\n  computed: {:?}",
            manifest.digests,
            computed
        ));
    }
    Ok(())
}

fn check_reference_and_admission(manifest: &Manifest) -> Result<()> {
    let reference = &manifest.reference;
    if reference.reference_transformers_version != REFERENCE_TRANSFORMERS_VERSION
        || reference.reference_seed != REFERENCE_SEED
        || reference.device != "cpu"
        || reference.dtype != "fp32"
    {
        return Err(perr!(
            "reference must be Transformers {REFERENCE_TRANSFORMERS_VERSION}, seed {REFERENCE_SEED}, cpu, fp32; found {reference:?}"
        ));
    }
    let admission = &manifest.admission;
    if admission.max_context_tokens != MAX_CONTEXT_TOKENS
        || admission.ane_resident_shapes_per_model != ANE_RESIDENT_SHAPES_PER_MODEL
        || admission.ane_resident_shapes_total != ANE_RESIDENT_SHAPES_TOTAL
        || admission.ane_shape_ladder != ANE_SHAPE_LADDER
    {
        return Err(perr!(
            "admission constants differ from the pinned values: {admission:?}"
        ));
    }
    if manifest.converter.rule != "v1" {
        return Err(perr!("converter rule must be `v1`"));
    }
    Ok(())
}

fn check_rotations(manifest: &Manifest) -> Result<()> {
    let names: Vec<&String> = manifest.rotations.keys().collect();
    if names != [MODERNBERT_ROTATION] {
        return Err(perr!(
            "rotations must be exactly [{MODERNBERT_ROTATION}], found {names:?}"
        ));
    }
    let rotation = &manifest.rotations[MODERNBERT_ROTATION];
    if rotation.sha256 != MODERNBERT_ROTATION_SHA256 || rotation.sign_seed != 0 {
        return Err(perr!(
            "rotation `{MODERNBERT_ROTATION}` entry differs from the pinned matrix: {rotation:?}"
        ));
    }
    hadamard::regenerate_checked(MODERNBERT_ROTATION, rotation)?;
    Ok(())
}

fn read_json(path: &Path) -> Result<(Vec<u8>, Value)> {
    let bytes = std::fs::read(path).map_err(|error| perr!("read {}: {error}", path.display()))?;
    let value = serde_json::from_slice(&bytes)
        .map_err(|error| perr!("parse {}: {error}", path.display()))?;
    Ok((bytes, value))
}

fn check_pinned_copy(model: &Model, file: &str, bytes: &[u8]) -> Result<()> {
    let pinned = model
        .files
        .get(file)
        .ok_or_else(|| perr!("no pinned digest for `{file}`"))?;
    let digest = sha256_hex(bytes);
    if &digest != pinned {
        return Err(perr!(
            "committed copy of `{file}` has SHA-256 {digest}, manifest pins {pinned}"
        ));
    }
    Ok(())
}

fn check_model(slug: &str, model: &Model, parity_dir: &Path) -> Result<()> {
    if model.hf_revision.len() != 40 || !model.hf_revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(perr!(
            "hf_revision `{}` is not a 40-hex commit",
            model.hf_revision
        ));
    }
    for required in ["config.json", "model.safetensors", "tokenizer.json"] {
        if !model.files.contains_key(required) {
            return Err(perr!("no pinned digest for `{required}`"));
        }
    }
    for (file, digest) in &model.files {
        if !is_sha256_hex(digest) {
            return Err(perr!(
                "pinned digest for `{file}` is not lowercase SHA-256 hex"
            ));
        }
    }
    if !is_sha256_hex(&model.tensor_index_sha256) {
        return Err(perr!("tensor_index_sha256 is not lowercase SHA-256 hex"));
    }
    if model.checkpoint_digest != format!("sha256:{}", model.files["model.safetensors"]) {
        return Err(perr!(
            "checkpoint_digest must be `sha256:` + the model.safetensors digest"
        ));
    }
    if model.tokenizer_digest != format!("sha256:{}", model.files["tokenizer.json"]) {
        return Err(perr!(
            "tokenizer_digest must be `sha256:` + the tokenizer.json digest"
        ));
    }
    check_schema(slug, &model.architecture)?;

    // The committed copies are digest-bound to the checkpoint, so checking
    // against them is checking against the pinned checkpoint.
    let dir = parity_dir.join("checkpoints").join(slug);
    let (config_bytes, config) = read_json(&dir.join("config.json"))?;
    check_pinned_copy(model, "config.json", &config_bytes)?;
    let tokenizer_config = if model.files.contains_key("tokenizer_config.json") {
        let (bytes, value) = read_json(&dir.join("tokenizer_config.json"))?;
        check_pinned_copy(model, "tokenizer_config.json", &bytes)?;
        Some(value)
    } else {
        None
    };
    check_against_config(
        slug,
        &model.architecture,
        &config,
        tokenizer_config.as_ref(),
    )?;

    let index_path = dir.join("tensor-index.json");
    let index_bytes = std::fs::read(&index_path)
        .map_err(|error| perr!("read {}: {error}", index_path.display()))?;
    if sha256_hex(&index_bytes) != model.tensor_index_sha256 {
        return Err(perr!(
            "committed tensor index digest differs from tensor_index_sha256"
        ));
    }
    let index: BTreeMap<String, Vec<u64>> = parse_header_json(&index_bytes)?
        .tensors
        .into_iter()
        .map(|(name, info)| (name, info.shape))
        .collect();
    check_tensor_index(model, &index)?;
    check_output(slug, model)?;
    check_grammar(model, &config, parity_dir)?;
    check_head(model, &config, &index)?;
    Ok(())
}

/// The tensor index must hold exactly the tensors the architecture
/// parameters imply, with the same shapes.
pub fn check_tensor_index(model: &Model, index: &BTreeMap<String, Vec<u64>>) -> Result<()> {
    let expected = expected_tensors(model)?;
    for (name, shape) in &expected {
        match index.get(name) {
            None => return Err(perr!("tensor index lacks `{name}`")),
            Some(actual) if actual != shape => {
                return Err(perr!(
                    "tensor `{name}` has shape {actual:?}, parameters imply {shape:?}"
                ))
            }
            Some(_) => {}
        }
    }
    if let Some(extra) = index.keys().find(|name| !expected.contains_key(*name)) {
        return Err(perr!(
            "tensor index has `{extra}`, which the parameters do not account for"
        ));
    }
    Ok(())
}

/// Output shape, pooling and normalization per model: the embedders return a
/// hidden-width L2-normalized vector (gte by CLS, Qwen3 by last non-pad
/// position); the rerankers return one unnormalized score (gte by masked
/// mean, Qwen3 by last non-pad position).
fn check_output(slug: &str, model: &Model) -> Result<()> {
    let hidden = model.architecture.int("hidden_size")? as u32;
    let (pooling, dimension, normalization) = match (slug, model.operation) {
        ("gte-modernbert-base", Operation::Embed) => (Pooling::Cls, hidden, Normalization::L2),
        ("qwen3-embedding-0.6b", Operation::Embed) => {
            (Pooling::LastNonPad, hidden, Normalization::L2)
        }
        ("gte-reranker-modernbert-base", Operation::Rerank) => {
            (Pooling::MaskedMean, 1, Normalization::None)
        }
        ("qwen3-reranker-0.6b", Operation::Rerank) => (Pooling::LastNonPad, 1, Normalization::None),
        _ => {
            return Err(perr!(
                "operation {:?} is not this model's operation",
                model.operation
            ))
        }
    };
    if model.grammar.pooling != pooling
        || model.output.dimension != dimension
        || model.output.normalization != normalization
    {
        return Err(perr!(
            "output must be pooling {pooling:?}, dimension {dimension}, normalization {normalization:?}"
        ));
    }
    Ok(())
}

fn config_u64(config: &Value, key: &str) -> Result<u64> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| perr!("pinned config.json has no integer `{key}`"))
}

fn check_grammar(model: &Model, config: &Value, parity_dir: &Path) -> Result<()> {
    let grammar: &Grammar = &model.grammar;
    let arch = &model.architecture;
    if u64::from(grammar.pad.id) != arch.int("pad_token_id")? {
        return Err(perr!(
            "grammar pad id {} differs from pad_token_id",
            grammar.pad.id
        ));
    }
    let roles: BTreeSet<&str> = grammar.special_tokens.keys().map(String::as_str).collect();
    match (arch.family, model.operation) {
        (Family::Modernbert, operation) => {
            let kind = if operation == Operation::Embed {
                GrammarKind::SingleSequence
            } else {
                GrammarKind::Pair
            };
            let readout = if operation == Operation::Embed {
                ReadoutKind::PooledHiddenState
            } else {
                ReadoutKind::SigmoidClassifierLogit
            };
            if grammar.kind != kind || grammar.template.is_some() || grammar.readout.kind != readout
            {
                return Err(perr!("ModernBERT {operation:?} grammar must be {kind:?} with readout {readout:?} and no template"));
            }
            if roles != BTreeSet::from(["cls", "sep"]) {
                return Err(perr!(
                    "ModernBERT grammar special tokens must be exactly cls and sep"
                ));
            }
            if u64::from(grammar.special_tokens["cls"].id) != config_u64(config, "cls_token_id")?
                || u64::from(grammar.special_tokens["sep"].id)
                    != config_u64(config, "sep_token_id")?
            {
                return Err(perr!(
                    "ModernBERT cls/sep ids differ from the pinned config.json"
                ));
            }
            if grammar.terminal_tokens != [grammar.special_tokens["sep"].clone()] {
                return Err(perr!("ModernBERT inputs end in exactly one sep token"));
            }
        }
        (Family::Qwen3, Operation::Embed) => {
            if grammar.kind != GrammarKind::SingleSequence
                || grammar.template.is_some()
                || grammar.readout.kind != ReadoutKind::PooledHiddenState
            {
                return Err(perr!(
                    "Qwen3 embed grammar must be a single sequence with a pooled readout"
                ));
            }
            if roles != BTreeSet::from(["eos"]) {
                return Err(perr!(
                    "Qwen3 embed grammar special tokens must be exactly eos"
                ));
            }
            if u64::from(grammar.special_tokens["eos"].id) != arch.int("eos_token_id")? {
                return Err(perr!("Qwen3 embed eos differs from eos_token_id"));
            }
            if grammar.terminal_tokens != [grammar.special_tokens["eos"].clone()] {
                return Err(perr!("Qwen3 embed inputs end in exactly one eos token"));
            }
        }
        (Family::Qwen3, Operation::Rerank) => {
            if grammar.kind != GrammarKind::Template
                || grammar.readout.kind != ReadoutKind::YesNoTwoWaySoftmax
            {
                return Err(perr!(
                    "Qwen3 rerank grammar must be a template with a yes/no readout"
                ));
            }
            if !grammar.terminal_tokens.is_empty() {
                return Err(perr!("Qwen3 rerank appends no terminal token"));
            }
            let template = grammar
                .template
                .as_ref()
                .ok_or_else(|| perr!("Qwen3 rerank grammar has no template"))?;
            for (role, token) in &grammar.special_tokens {
                if !template.prefix.contains(&token.text) && !template.suffix.contains(&token.text)
                {
                    return Err(perr!(
                        "special token `{role}` does not occur in the template"
                    ));
                }
            }
            let oracle_path = parity_dir.join(&template.oracle.path);
            let readme = std::fs::read(&oracle_path)
                .map_err(|error| perr!("read {}: {error}", oracle_path.display()))?;
            if sha256_hex(&readme) != template.oracle.sha256
                || model.files.get("README.md") != Some(&template.oracle.sha256)
            {
                return Err(perr!(
                    "template oracle digest differs from the pinned one or from the checkpoint's README.md"
                ));
            }
            let readme =
                String::from_utf8(readme).map_err(|_| perr!("template oracle is not UTF-8"))?;
            check_template(template, &readme)?;
            let (yes, no) = match (&grammar.readout.yes, &grammar.readout.no) {
                (Some(yes), Some(no)) => (yes, no),
                _ => {
                    return Err(perr!(
                        "qwen_readout_mismatch: yes and no ids must both be recorded"
                    ))
                }
            };
            if yes.text != "yes" || no.text != "no" || yes.id == no.id {
                return Err(perr!(
                    "qwen_readout_mismatch: readout must be distinct `yes` and `no` ids"
                ));
            }
            let vocab = arch.int("vocab_size")?;
            if u64::from(yes.id) >= vocab || u64::from(no.id) >= vocab {
                return Err(perr!(
                    "qwen_readout_mismatch: yes/no ids outside the vocabulary"
                ));
            }
        }
    }
    if model.operation == Operation::Embed
        && (grammar.readout.yes.is_some() || grammar.readout.weight.is_some())
    {
        return Err(perr!(
            "embed readout carries no yes/no ids or readout weight"
        ));
    }
    Ok(())
}

/// Rerank head checks against the pinned config and tensor index.
pub fn check_head(model: &Model, config: &Value, index: &BTreeMap<String, Vec<u64>>) -> Result<()> {
    let head: &Head = match (model.operation, &model.head) {
        (Operation::Embed, None) => return Ok(()),
        (Operation::Embed, Some(_)) => return Err(perr!("embed models have no rerank head")),
        (Operation::Rerank, None) => return Err(perr!("rerank model without a head")),
        (Operation::Rerank, Some(head)) => head,
    };
    let arch = &model.architecture;
    let hidden = arch.int("hidden_size")?;
    let vocab = arch.int("vocab_size")?;
    let tensor = |key: String, shape: Vec<u64>| crate::manifest::HeadTensor { key, shape };
    let (pooling, stages, tensors, forbidden, norm_eps) = match arch.family {
        Family::Modernbert => {
            let pooling = match config.get("classifier_pooling").and_then(Value::as_str) {
                Some("mean") => Pooling::MaskedMean,
                Some("cls") => Pooling::Cls,
                other => return Err(perr!("classifier_pooling {other:?} is not supported")),
            };
            let activation = config
                .get("classifier_activation")
                .and_then(Value::as_str)
                .ok_or_else(|| perr!("pinned config.json has no classifier_activation"))?;
            let labels = config
                .get("id2label")
                .and_then(Value::as_object)
                .map(|labels| labels.len() as u64)
                .ok_or_else(|| perr!("pinned config.json has no id2label"))?;
            let classifier_bias = config
                .get("classifier_bias")
                .and_then(Value::as_bool)
                .ok_or_else(|| perr!("pinned config.json has no classifier_bias"))?;
            if classifier_bias || arch.boolean("norm_bias")? {
                return Err(perr!(
                    "head biases switched on in config.json are not supported"
                ));
            }
            let tensors = BTreeMap::from([
                (
                    "dense".to_string(),
                    tensor("head.dense.weight".into(), vec![hidden, hidden]),
                ),
                (
                    "norm".to_string(),
                    tensor("head.norm.weight".into(), vec![hidden]),
                ),
                (
                    "classifier".to_string(),
                    tensor("classifier.weight".into(), vec![labels, hidden]),
                ),
                (
                    "classifier_bias".to_string(),
                    tensor("classifier.bias".into(), vec![labels]),
                ),
            ]);
            let stages = vec![
                "dense".to_string(),
                format!("activation:{activation}"),
                "layernorm".to_string(),
                "classifier".to_string(),
            ];
            let forbidden = vec!["head.dense.bias".to_string(), "head.norm.bias".to_string()];
            (
                pooling,
                stages,
                tensors,
                forbidden,
                Some(arch.float("norm_eps")?),
            )
        }
        Family::Qwen3 => {
            // The yes/no readout rows come from lm_head.weight, or from the
            // token embedding when the checkpoint ties the two.
            let (key, forbidden) = if arch.boolean("tie_word_embeddings")? {
                (
                    format!("{}embed_tokens.weight", model.tensor_prefix),
                    vec!["lm_head.weight".to_string()],
                )
            } else {
                ("lm_head.weight".to_string(), vec![])
            };
            if model.grammar.readout.weight.as_deref() != Some(key.as_str()) {
                return Err(perr!(
                    "qwen_readout_mismatch: readout weight must be `{key}`"
                ));
            }
            let tensors =
                BTreeMap::from([("readout".to_string(), tensor(key, vec![vocab, hidden]))]);
            (
                Pooling::LastNonPad,
                vec!["readout_rows".to_string()],
                tensors,
                forbidden,
                None,
            )
        }
    };
    let required: Vec<String> = {
        let mut keys: Vec<String> = tensors.values().map(|t| t.key.clone()).collect();
        keys.sort();
        keys
    };
    let expected = Head {
        pooling,
        stages,
        tensors,
        required_tensor_keys: required,
        forbidden_tensor_keys: forbidden,
        norm_eps,
    };
    if &expected != head {
        return Err(perr!(
            "rerank head disagrees with the pinned checkpoint:\n  manifest: {}\n  derived:  {}",
            serde_json::to_string(head).expect("head serializes"),
            serde_json::to_string(&expected).expect("head serializes")
        ));
    }
    if head.pooling != model.grammar.pooling {
        return Err(perr!("head pooling differs from the grammar pooling"));
    }
    for tensor in head.tensors.values() {
        match index.get(&tensor.key) {
            None => {
                return Err(perr!(
                    "head_tensor_missing: `{}` is not in the tensor index",
                    tensor.key
                ))
            }
            Some(shape) if shape != &tensor.shape => {
                return Err(perr!(
                    "head tensor `{}` has shape {shape:?}, manifest says {:?}",
                    tensor.key,
                    tensor.shape
                ))
            }
            Some(_) => {}
        }
    }
    if let Some(present) = head
        .forbidden_tensor_keys
        .iter()
        .find(|key| index.contains_key(*key))
    {
        return Err(perr!("tensor `{present}` is present but must be absent"));
    }
    Ok(())
}

/// Check the committed certification inputs beside the manifest: the machine
/// registry, every row's module config fixture against the module defaults
/// (read from the module source in this checkout), the release asset
/// inventory and the preload regression inputs.
pub fn validate_inventory(parity_dir: &Path) -> Result<()> {
    use crate::inventory::{
        check_machine_registry, check_release_assets, check_row_fixture, load_json,
        module_defaults, MachineRegistry, ReleaseAssets,
    };
    use crate::preload::{check_preload, PreloadInputs, PRELOAD_MODEL_IDS};

    let registry: MachineRegistry = load_json(&parity_dir.join("machines.json"))?;
    check_machine_registry(&registry, parity_dir)?;
    let lib_rs_path = crate::repo_root().join("crates/synapse-module/src/lib.rs");
    let lib_rs = std::fs::read_to_string(&lib_rs_path)
        .map_err(|error| perr!("read {}: {error}", lib_rs_path.display()))?;
    let defaults = module_defaults(&lib_rs)?;
    for (id, row) in &registry.rows {
        let fixture: Value = load_json(&parity_dir.join(&row.module_config))?;
        check_row_fixture(id, &fixture, &defaults)?;
    }
    let assets: ReleaseAssets = load_json(&parity_dir.join("release-assets.json"))?;
    check_release_assets(&assets, &registry)?;
    for model_id in PRELOAD_MODEL_IDS {
        let inputs: PreloadInputs =
            load_json(&parity_dir.join("preload").join(format!("{model_id}.json")))?;
        if inputs.model_id != model_id {
            return Err(perr!(
                "preload/{model_id}.json names model `{}`",
                inputs.model_id
            ));
        }
        for (section, values) in [("inline", &inputs.inline), ("jobs", &inputs.jobs)] {
            if values != &defaults[section] {
                return Err(perr!(
                    "preload/{model_id}.json `{section}` differs from the module defaults"
                ));
            }
        }
        check_preload(&inputs)?;
    }
    Ok(())
}

fn check_profiles(manifest: &Manifest) -> Result<()> {
    let mut expected_ids = BTreeSet::new();
    for slug in MODEL_SLUGS {
        for lane in Lane::ALL {
            expected_ids.insert(profile_id(slug, lane));
        }
    }
    let ids: BTreeSet<String> = manifest.profiles.keys().cloned().collect();
    if ids != expected_ids {
        return Err(perr!(
            "profiles must be exactly every <slug>.<lane identity>; found {ids:?}"
        ));
    }
    for (id, profile) in &manifest.profiles {
        if *id != profile_id(&profile.model, profile.lane) {
            return Err(perr!(
                "profile `{id}` names model `{}` on lane `{}`",
                profile.model,
                profile.lane.as_str()
            ));
        }
        let model = manifest.model(&profile.model)?;

        match (&profile.converted_package_digest, profile.lane.is_worker()) {
            (Some(digest), true) => {
                let hex = digest.strip_prefix("sha256:").ok_or_else(|| {
                    perr!("profile `{id}` package digest lacks the sha256: prefix")
                })?;
                if !is_sha256_hex(hex) {
                    return Err(perr!("profile `{id}` package digest is not SHA-256 hex"));
                }
            }
            (None, false) => {}
            (None, true) => {
                return Err(perr!(
                    "worker profile `{id}` has no converted_package_digest"
                ))
            }
            (Some(_), false) => {
                return Err(perr!(
                    "in-process profile `{id}` must not carry a converted_package_digest"
                ))
            }
        }
        let cuda = (
            profile.cuda_min_driver_api,
            profile.cuda_min_compute_major,
            profile.cuda_min_compute_minor,
        );
        match (profile.lane, cuda) {
            (Lane::OwnedCuda, (Some(driver), Some(major), Some(minor)))
                if driver > 0 && major > 0 && minor < 10 => {}
            (Lane::OwnedCuda, _) => {
                return Err(perr!("CUDA profile `{id}` lacks valid CUDA floor keys"))
            }
            (_, (None, None, None)) => {}
            _ => return Err(perr!("non-CUDA profile `{id}` carries CUDA floor keys")),
        }
        check_profile(id, &profile.model, model, profile)?;
        let vulkan = (
            profile.vulkan_sub_batch_max_tokens,
            profile.vulkan_min_storage_buffer_range,
            profile.vulkan_min_device_local_bytes,
        );
        match (profile.lane, vulkan) {
            (Lane::OwnedVulkan, (Some(sub_batch), Some(range), Some(local))) => {
                if sub_batch != VULKAN_SUB_BATCH_MAX_TOKENS {
                    return Err(perr!("profile `{id}` vulkan_sub_batch_max_tokens must be {VULKAN_SUB_BATCH_MAX_TOKENS}"));
                }
                let floors = vulkan_floors(
                    model,
                    profile.storage_dtype,
                    &profile.fp32_tensors,
                    u64::from(manifest.admission.max_context_tokens),
                    FLOOR_SEQUENCES,
                    u64::from(sub_batch),
                )?;
                if floors.min_storage_buffer_range != range
                    || floors.min_device_local_bytes != local
                {
                    return Err(perr!(
                        "profile `{id}` Vulkan floors ({range}, {local}) differ from the sizing function ({}, {})",
                        floors.min_storage_buffer_range,
                        floors.min_device_local_bytes
                    ));
                }
            }
            (Lane::OwnedVulkan, _) => {
                return Err(perr!("Vulkan profile `{id}` lacks a Vulkan floor key"))
            }
            (_, (None, None, None)) => {}
            _ => return Err(perr!("non-Vulkan profile `{id}` carries Vulkan keys")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod cuda_floor_tests {
    use super::*;
    #[test]
    fn cuda_floor_fields_are_required_and_exclusive() {
        let manifest = Manifest::from_slice(include_bytes!("../models.json")).unwrap();
        check_profiles(&manifest).unwrap();
        for field in 0..3 {
            let mut missing = manifest.clone();
            let p = missing
                .profiles
                .get_mut("gte-modernbert-base.owned-cuda")
                .unwrap();
            match field {
                0 => p.cuda_min_driver_api = None,
                1 => p.cuda_min_compute_major = None,
                _ => p.cuda_min_compute_minor = None,
            };
            assert!(check_profiles(&missing)
                .unwrap_err()
                .to_string()
                .contains("CUDA floor keys"));
            let mut extra = manifest.clone();
            let p = extra
                .profiles
                .get_mut("gte-modernbert-base.owned-vulkan")
                .unwrap();
            match field {
                0 => p.cuda_min_driver_api = Some(13020),
                1 => p.cuda_min_compute_major = Some(7),
                _ => p.cuda_min_compute_minor = Some(5),
            };
            assert!(check_profiles(&extra)
                .unwrap_err()
                .to_string()
                .contains("non-CUDA"));
        }
    }
}
