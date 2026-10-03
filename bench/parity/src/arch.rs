//! Per-architecture parameter tables.
//!
//! Every engine reads its architecture from the manifest, never from a
//! checkpoint's `config.json` and never from an engine default. This module
//! fixes, per family, exactly which parameters the manifest carries and where
//! in the pinned checkpoint each one is checked:
//!
//! - `Config(key)`: the value of `key` in `config.json`;
//! - `PadTokenFromTokenizerConfig`: the id of `tokenizer_config.json`'s
//!   `pad_token` (Qwen3 checkpoints carry no `pad_token_id` in `config.json`);
//! - `Class`: a fact about the checkpoint's `architectures[0]` class that has
//!   no `config.json` key (the Qwen3 causal mask, the norm type). These come
//!   from `class_table`, which refuses a class it does not know.
//!
//! A required value that is missing from its source, or that disagrees with
//! the manifest, is an error.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::manifest::{Architecture, Family, Model};
use crate::{perr, Result};

/// JSON type a parameter must have in the manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Int,
    Float,
    Bool,
    Str,
    Null,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Config(&'static str),
    PadTokenFromTokenizerConfig,
    Class,
}

#[derive(Clone, Copy, Debug)]
pub struct ParamSpec {
    pub name: &'static str,
    pub kind: Kind,
    pub source: Source,
}

const fn spec(name: &'static str, kind: Kind, source: Source) -> ParamSpec {
    ParamSpec { name, kind, source }
}

/// ModernBERT parameters: the common set plus the local attention window,
/// the global-attention layer cadence, the global and local RoPE bases and
/// the LayerNorm settings.
pub const MODERNBERT_PARAMS: &[ParamSpec] = &[
    spec(
        "num_hidden_layers",
        Kind::Int,
        Source::Config("num_hidden_layers"),
    ),
    spec("hidden_size", Kind::Int, Source::Config("hidden_size")),
    spec(
        "num_attention_heads",
        Kind::Int,
        Source::Config("num_attention_heads"),
    ),
    spec(
        "intermediate_size",
        Kind::Int,
        Source::Config("intermediate_size"),
    ),
    spec("vocab_size", Kind::Int, Source::Config("vocab_size")),
    spec("pad_token_id", Kind::Int, Source::Config("pad_token_id")),
    spec("eos_token_id", Kind::Int, Source::Config("eos_token_id")),
    spec("norm_type", Kind::Str, Source::Class),
    spec("norm_eps", Kind::Float, Source::Config("norm_eps")),
    spec("norm_bias", Kind::Bool, Source::Config("norm_bias")),
    spec("activation", Kind::Str, Source::Config("hidden_activation")),
    spec("tie_word_embeddings", Kind::Bool, Source::Class),
    spec("attention_mask", Kind::Str, Source::Class),
    spec("mlp_kind", Kind::Str, Source::Class),
    spec(
        "attention_bias",
        Kind::Bool,
        Source::Config("attention_bias"),
    ),
    spec("mlp_bias", Kind::Bool, Source::Config("mlp_bias")),
    spec(
        "max_position_embeddings",
        Kind::Int,
        Source::Config("max_position_embeddings"),
    ),
    spec(
        "local_attention",
        Kind::Int,
        Source::Config("local_attention"),
    ),
    spec(
        "global_attn_every_n_layers",
        Kind::Int,
        Source::Config("global_attn_every_n_layers"),
    ),
    spec(
        "global_rope_theta",
        Kind::Float,
        Source::Config("global_rope_theta"),
    ),
    spec(
        "local_rope_theta",
        Kind::Float,
        Source::Config("local_rope_theta"),
    ),
];

/// Qwen3 parameters: the common set plus key/value heads, head dimension,
/// the causal mask and the single RoPE base.
pub const QWEN3_PARAMS: &[ParamSpec] = &[
    spec(
        "num_hidden_layers",
        Kind::Int,
        Source::Config("num_hidden_layers"),
    ),
    spec("hidden_size", Kind::Int, Source::Config("hidden_size")),
    spec(
        "num_attention_heads",
        Kind::Int,
        Source::Config("num_attention_heads"),
    ),
    spec(
        "num_key_value_heads",
        Kind::Int,
        Source::Config("num_key_value_heads"),
    ),
    spec("head_dim", Kind::Int, Source::Config("head_dim")),
    spec(
        "intermediate_size",
        Kind::Int,
        Source::Config("intermediate_size"),
    ),
    spec("vocab_size", Kind::Int, Source::Config("vocab_size")),
    spec(
        "pad_token_id",
        Kind::Int,
        Source::PadTokenFromTokenizerConfig,
    ),
    spec("eos_token_id", Kind::Int, Source::Config("eos_token_id")),
    spec("norm_type", Kind::Str, Source::Class),
    spec("norm_eps", Kind::Float, Source::Config("rms_norm_eps")),
    spec("activation", Kind::Str, Source::Config("hidden_act")),
    spec(
        "tie_word_embeddings",
        Kind::Bool,
        Source::Config("tie_word_embeddings"),
    ),
    spec("attention_mask", Kind::Str, Source::Class),
    spec("mlp_kind", Kind::Str, Source::Class),
    spec("qk_norm", Kind::Bool, Source::Class),
    spec(
        "attention_bias",
        Kind::Bool,
        Source::Config("attention_bias"),
    ),
    spec(
        "max_position_embeddings",
        Kind::Int,
        Source::Config("max_position_embeddings"),
    ),
    spec("rope_theta", Kind::Float, Source::Config("rope_theta")),
    spec("rope_scaling", Kind::Null, Source::Config("rope_scaling")),
    spec(
        "use_sliding_window",
        Kind::Bool,
        Source::Config("use_sliding_window"),
    ),
];

pub fn params_for(family: Family) -> &'static [ParamSpec] {
    match family {
        Family::Modernbert => MODERNBERT_PARAMS,
        Family::Qwen3 => QWEN3_PARAMS,
    }
}

/// Facts fixed by a checkpoint's architecture class.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassInfo {
    pub family: Family,
    /// Values for every `Source::Class` parameter of the family.
    pub values: BTreeMap<&'static str, Value>,
    /// Whether the class carries the ModernBERT sequence-classification head.
    pub has_classification_head: bool,
}

/// The fixed architectures table. An architecture class not listed here is
/// refused: its mask, norm and MLP shape are unknown, and no default may be
/// assumed for them.
pub fn class_table(class: &str) -> Result<ClassInfo> {
    let modernbert = |head: bool| ClassInfo {
        family: Family::Modernbert,
        values: BTreeMap::from([
            ("norm_type", json!("layernorm")),
            // Neither ModernBERT class used here instantiates an output
            // embedding, so there is nothing to tie; the tensor index check
            // also refuses any `decoder` or `lm_head` tensor.
            ("tie_word_embeddings", json!(false)),
            ("attention_mask", json!("bidirectional")),
            ("mlp_kind", json!("geglu")),
        ]),
        has_classification_head: head,
    };
    match class {
        "ModernBertModel" => Ok(modernbert(false)),
        "ModernBertForSequenceClassification" => Ok(modernbert(true)),
        "Qwen3ForCausalLM" => Ok(ClassInfo {
            family: Family::Qwen3,
            values: BTreeMap::from([
                ("norm_type", json!("rmsnorm")),
                ("attention_mask", json!("causal")),
                ("mlp_kind", json!("swiglu")),
                ("qk_norm", json!(true)),
            ]),
            has_classification_head: false,
        }),
        other => Err(perr!(
            "unknown architecture `{other}`: it is not in the fixed architectures table, so its attention mask and layer layout cannot be checked"
        )),
    }
}

fn kind_matches(kind: Kind, value: &Value) -> bool {
    match kind {
        Kind::Int => value.is_u64(),
        // Floats are written as JSON floats in the manifest so the canonical
        // bytes do not depend on how the checkpoint spelled the number.
        Kind::Float => value.is_f64(),
        Kind::Bool => value.is_boolean(),
        Kind::Str => value.is_string(),
        Kind::Null => value.is_null(),
    }
}

/// Schema-level check of an architecture block: the class is known, the
/// family matches it, the parameter names are exactly the family's set, each
/// value has its declared JSON type, and every class-sourced value equals
/// the table. Needs no checkpoint.
pub fn check_schema(slug: &str, arch: &Architecture) -> Result<ClassInfo> {
    let info = class_table(&arch.class).map_err(|error| perr!("{slug}: {error}"))?;
    if info.family != arch.family {
        return Err(perr!(
            "{slug}: family {:?} does not match class `{}`",
            arch.family,
            arch.class
        ));
    }
    let specs = params_for(arch.family);
    let expected: BTreeSet<&str> = specs.iter().map(|spec| spec.name).collect();
    let present: BTreeSet<&str> = arch.params.keys().map(String::as_str).collect();
    if let Some(missing) = expected.difference(&present).next() {
        return Err(perr!(
            "{slug}: architecture parameter `{missing}` is missing"
        ));
    }
    if let Some(extra) = present.difference(&expected).next() {
        return Err(perr!(
            "{slug}: architecture parameter `{extra}` is not in the {:?} parameter table",
            arch.family
        ));
    }
    for spec in specs {
        let value = &arch.params[spec.name];
        if !kind_matches(spec.kind, value) {
            return Err(perr!(
                "{slug}: architecture parameter `{}` = {value} is not of kind {:?}",
                spec.name,
                spec.kind
            ));
        }
        if spec.source == Source::Class {
            let table = &info.values[spec.name];
            if table != value {
                return Err(perr!(
                    "{slug}: architecture parameter `{}` = {value} disagrees with the architectures table value {table} for `{}`",
                    spec.name,
                    arch.class
                ));
            }
        }
    }
    Ok(info)
}

fn values_equal(kind: Kind, manifest: &Value, source: &Value) -> bool {
    match kind {
        Kind::Float => match (manifest.as_f64(), source.as_f64()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        },
        _ => manifest == source,
    }
}

/// Check every architecture parameter against the pinned checkpoint's
/// `config.json` (and `tokenizer_config.json` for the Qwen3 pad id). Fails on
/// a missing or disagreeing value, and on an unknown architecture class.
pub fn check_against_config(
    slug: &str,
    arch: &Architecture,
    config: &Value,
    tokenizer_config: Option<&Value>,
) -> Result<()> {
    let classes = config
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or_else(|| perr!("{slug}: config.json has no `architectures` list"))?;
    let [class] = classes.as_slice() else {
        return Err(perr!(
            "{slug}: config.json lists {} architectures; exactly one is required",
            classes.len()
        ));
    };
    let class = class
        .as_str()
        .ok_or_else(|| perr!("{slug}: config.json `architectures[0]` is not a string"))?;
    // The class decides the family and every class-sourced value (the Qwen3
    // causal mask among them), so an unknown class fails here.
    class_table(class).map_err(|error| perr!("{slug}: {error}"))?;
    if class != arch.class {
        return Err(perr!(
            "{slug}: manifest class `{}` disagrees with config.json architectures[0] `{class}`",
            arch.class
        ));
    }
    check_schema(slug, arch)?;
    for spec in params_for(arch.family) {
        let manifest_value = &arch.params[spec.name];
        let source_value = match spec.source {
            Source::Class => continue,
            Source::Config(key) => config.get(key).cloned().ok_or_else(|| {
                perr!(
                    "{slug}: architecture parameter `{}` has no `{key}` in the pinned config.json",
                    spec.name
                )
            })?,
            Source::PadTokenFromTokenizerConfig => {
                let tokenizer_config = tokenizer_config.ok_or_else(|| {
                    perr!(
                        "{slug}: `{}` comes from tokenizer_config.json, which the checkpoint lacks",
                        spec.name
                    )
                })?;
                json!(pad_token_id(tokenizer_config)
                    .map_err(|error| perr!("{slug}: `{}`: {error}", spec.name))?)
            }
        };
        if !values_equal(spec.kind, manifest_value, &source_value) {
            return Err(perr!(
                "{slug}: architecture parameter `{}` = {manifest_value} disagrees with the pinned checkpoint value {source_value}",
                spec.name
            ));
        }
    }
    Ok(())
}

/// Id of `pad_token` per `tokenizer_config.json`'s `added_tokens_decoder`.
pub fn pad_token_id(tokenizer_config: &Value) -> Result<u64> {
    let pad = tokenizer_config
        .get("pad_token")
        .and_then(|value| value.as_str().or_else(|| value.get("content")?.as_str()))
        .ok_or_else(|| perr!("tokenizer_config.json has no `pad_token`"))?;
    let decoder = tokenizer_config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .ok_or_else(|| perr!("tokenizer_config.json has no `added_tokens_decoder`"))?;
    let mut ids = decoder.iter().filter_map(|(id, token)| {
        (token.get("content").and_then(Value::as_str) == Some(pad)).then_some(id)
    });
    let id = ids
        .next()
        .ok_or_else(|| perr!("pad token `{pad}` is not in `added_tokens_decoder`"))?;
    if ids.next().is_some() {
        return Err(perr!(
            "pad token `{pad}` appears twice in `added_tokens_decoder`"
        ));
    }
    id.parse()
        .map_err(|_| perr!("`added_tokens_decoder` key `{id}` is not an integer"))
}

/// Every tensor (name and shape) the checkpoint must hold, derived from the
/// manifest's architecture parameters. The generator requires the pinned
/// tensor index to equal this set exactly (`validate::check_tensor_index`),
/// which ties layer counts, widths and head layout to the real weights.
/// `vulkan::buffer_plan` sizes weight buffers from the same list.
pub fn expected_tensors(model: &Model) -> Result<BTreeMap<String, Vec<u64>>> {
    let arch = &model.architecture;
    let info = class_table(&arch.class)?;
    let p = model.tensor_prefix.as_str();
    let layers = arch.int("num_hidden_layers")?;
    let hidden = arch.int("hidden_size")?;
    let intermediate = arch.int("intermediate_size")?;
    let vocab = arch.int("vocab_size")?;
    let mut out = BTreeMap::new();
    let mut put = |name: String, shape: Vec<u64>| {
        out.insert(name, shape);
    };
    match arch.family {
        Family::Modernbert => {
            for flag in ["norm_bias", "attention_bias", "mlp_bias"] {
                if arch.boolean(flag)? {
                    return Err(perr!(
                        "ModernBERT `{flag}` = true is not supported by the tensor table"
                    ));
                }
            }
            put(
                format!("{p}embeddings.tok_embeddings.weight"),
                vec![vocab, hidden],
            );
            put(format!("{p}embeddings.norm.weight"), vec![hidden]);
            for layer in 0..layers {
                let l = format!("{p}layers.{layer}");
                // Layer 0 has no attention norm: the embedding norm feeds it.
                if layer > 0 {
                    put(format!("{l}.attn_norm.weight"), vec![hidden]);
                }
                put(format!("{l}.attn.Wqkv.weight"), vec![3 * hidden, hidden]);
                put(format!("{l}.attn.Wo.weight"), vec![hidden, hidden]);
                put(format!("{l}.mlp_norm.weight"), vec![hidden]);
                put(format!("{l}.mlp.Wi.weight"), vec![2 * intermediate, hidden]);
                put(format!("{l}.mlp.Wo.weight"), vec![hidden, intermediate]);
            }
            put(format!("{p}final_norm.weight"), vec![hidden]);
            if info.has_classification_head {
                let head = model
                    .head
                    .as_ref()
                    .ok_or_else(|| perr!("classification checkpoint without a head entry"))?;
                for tensor in head.tensors.values() {
                    put(tensor.key.clone(), tensor.shape.clone());
                }
            }
        }
        Family::Qwen3 => {
            if arch.boolean("attention_bias")? {
                return Err(perr!(
                    "Qwen3 `attention_bias` = true is not supported by the tensor table"
                ));
            }
            let heads = arch.int("num_attention_heads")?;
            let kv_heads = arch.int("num_key_value_heads")?;
            let head_dim = arch.int("head_dim")?;
            put(format!("{p}embed_tokens.weight"), vec![vocab, hidden]);
            for layer in 0..layers {
                let l = format!("{p}layers.{layer}");
                put(format!("{l}.input_layernorm.weight"), vec![hidden]);
                put(format!("{l}.post_attention_layernorm.weight"), vec![hidden]);
                put(
                    format!("{l}.self_attn.q_proj.weight"),
                    vec![heads * head_dim, hidden],
                );
                put(
                    format!("{l}.self_attn.k_proj.weight"),
                    vec![kv_heads * head_dim, hidden],
                );
                put(
                    format!("{l}.self_attn.v_proj.weight"),
                    vec![kv_heads * head_dim, hidden],
                );
                put(
                    format!("{l}.self_attn.o_proj.weight"),
                    vec![hidden, heads * head_dim],
                );
                put(format!("{l}.self_attn.q_norm.weight"), vec![head_dim]);
                put(format!("{l}.self_attn.k_norm.weight"), vec![head_dim]);
                put(
                    format!("{l}.mlp.gate_proj.weight"),
                    vec![intermediate, hidden],
                );
                put(
                    format!("{l}.mlp.up_proj.weight"),
                    vec![intermediate, hidden],
                );
                put(
                    format!("{l}.mlp.down_proj.weight"),
                    vec![hidden, intermediate],
                );
            }
            put(format!("{p}norm.weight"), vec![hidden]);
            if !arch.boolean("tie_word_embeddings")? {
                put("lm_head.weight".to_string(), vec![vocab, hidden]);
            }
        }
    }
    Ok(out)
}

/// Name of the token-embedding table in the checkpoint.
pub fn embedding_key(model: &Model) -> String {
    match model.architecture.family {
        Family::Modernbert => format!("{}embeddings.tok_embeddings.weight", model.tensor_prefix),
        Family::Qwen3 => format!("{}embed_tokens.weight", model.tensor_prefix),
    }
}

/// Name of the final norm's scale in the checkpoint.
pub fn final_norm_key(model: &Model) -> String {
    match model.architecture.family {
        Family::Modernbert => format!("{}final_norm.weight", model.tensor_prefix),
        Family::Qwen3 => format!("{}norm.weight", model.tensor_prefix),
    }
}
