//! Qwen3-ASR checkpoint facts. No serving lane is admitted by this module.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::json;

use crate::manifest::{Family, Model};
use crate::{perr, Result};

pub const SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];
pub const TOKENIZER_FILES: [&str; 3] = ["merges.txt", "tokenizer_config.json", "vocab.json"];

/// Aggregate identities bind file names as well as their pinned bytes.
pub fn file_map_digest(model: &Model, names: &[&str]) -> Result<String> {
    let files: BTreeMap<&str, &String> = names
        .iter()
        .map(|name| {
            model
                .files
                .get(*name)
                .map(|digest| (*name, digest))
                .ok_or_else(|| perr!("missing pinned file {name}"))
        })
        .collect::<Result<_>>()?;
    Ok(format!(
        "sha256:{}",
        crate::canonical::sha256_hex(&crate::canonical::canonical_bytes_of(&files))
    ))
}

/// Canonical merged headers retain shard-local offsets. Refuse duplicate tensor
/// names and a weight map that disagrees with the shard holding a tensor.
pub fn merged_header(dir: &Path) -> Result<Vec<u8>> {
    let map: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("model.safetensors.index.json"))
            .map_err(|e| perr!("read shard weight map: {e}"))?,
    )
    .map_err(|e| perr!("shard weight map: {e}"))?;
    let weights = map
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| perr!("missing shard weight_map"))?;
    let mut merged = serde_json::Map::new();
    for shard in SHARDS {
        let bytes = crate::safetensors::read_header_bytes(&dir.join(shard))?;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| perr!("shard header: {e}"))?;
        for (name, value) in header
            .as_object()
            .ok_or_else(|| perr!("shard header is not an object"))?
        {
            if name == "__metadata__" {
                if merged.get(name).is_some_and(|old| old != value) {
                    return Err(perr!("conflicting shard metadata"));
                }
            } else {
                if merged.contains_key(name) {
                    return Err(perr!("duplicate shard tensor {name}"));
                }
                if weights.get(name).and_then(serde_json::Value::as_str) != Some(shard) {
                    return Err(perr!("weight_map disagrees with shard for {name}"));
                }
            }
            merged.insert(name.clone(), value.clone());
        }
    }
    if weights.len()
        != merged
            .keys()
            .filter(|name| name.as_str() != "__metadata__")
            .count()
    {
        return Err(perr!("weight_map contains a tensor absent from the shards"));
    }
    Ok(crate::canonical::canonical_bytes(
        &serde_json::Value::Object(merged),
    ))
}

/// Exact independent shape inventory for the encoder plus the Qwen3 decoder.
pub fn expected_tensors(model: &Model) -> Result<BTreeMap<String, Vec<u64>>> {
    let arch = &model.architecture;
    let mut decoder = model.clone();
    decoder.architecture.family = Family::Qwen3;
    decoder.architecture.class = "Qwen3ForCausalLM".into();
    decoder
        .architecture
        .params
        .insert("attention_bias".into(), json!(false));
    let mut tensors = crate::arch::expected_tensors(&decoder)?;
    tensors.remove("lm_head.weight");
    // The official ASR class always constructs and stores the head, including
    // checkpoints whose nested text config declares tied input/output weights.
    tensors.insert(
        "thinker.lm_head.weight".into(),
        vec![arch.int("vocab_size")?, arch.int("hidden_size")?],
    );
    let h = arch.int("audio_hidden_size")?;
    let f = arch.int("audio_intermediate_size")?;
    let c = arch.int("conv_channels")?;
    let output = arch.int("audio_output_dim")?;
    if output != arch.int("hidden_size")? || h % arch.int("audio_num_heads")? != 0 {
        return Err(perr!("Qwen3-ASR encoder/decoder dimensions disagree"));
    }
    let mut put = |name: String, shape: Vec<u64>| {
        tensors.insert(format!("thinker.audio_tower.{name}"), shape);
    };
    for stage in 1..=3 {
        put(
            format!("conv2d{stage}.weight"),
            vec![c, if stage == 1 { 1 } else { c }, 3, 3],
        );
        put(format!("conv2d{stage}.bias"), vec![c]);
    }
    put(
        "conv_out.weight".into(),
        vec![h, c * arch.int("num_mel_bins")?.div_ceil(8)],
    );
    for layer in 0..arch.int("audio_num_layers")? {
        let l = format!("layers.{layer}");
        for norm in ["self_attn_layer_norm", "final_layer_norm"] {
            for suffix in ["weight", "bias"] {
                put(format!("{l}.{norm}.{suffix}"), vec![h]);
            }
        }
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            put(format!("{l}.self_attn.{projection}.weight"), vec![h, h]);
            put(format!("{l}.self_attn.{projection}.bias"), vec![h]);
        }
        for (name, a, b) in [("fc1", f, h), ("fc2", h, f)] {
            put(format!("{l}.{name}.weight"), vec![a, b]);
            put(format!("{l}.{name}.bias"), vec![a]);
        }
    }
    for suffix in ["weight", "bias"] {
        put(format!("ln_post.{suffix}"), vec![h]);
    }
    for (name, a) in [("proj1", h), ("proj2", output)] {
        put(format!("{name}.weight"), vec![a, h]);
        put(format!("{name}.bias"), vec![a]);
    }
    Ok(tensors)
}
