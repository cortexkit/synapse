//! Offline pin producer. Shares production code, but never loads an engine.
#![allow(dead_code)]

use anyhow::{ensure, Context, Result};
use serde_json::json;
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[path = "../../../../crates/synapse-owned-decode/owned-decode-routing/error.rs"]
pub mod decode_error;
#[path = "../../../../crates/synapse-owned-decode/owned-decode-routing/family.rs"]
pub mod decode_family;
#[path = "../../../../crates/synapse-owned-decode/owned-decode-routing/q8artifact.rs"]
pub mod q8artifact;
#[path = "../../../../crates/synapse-owned-decode/owned-decode-routing/q8asr.rs"]
pub mod q8asr;
#[path = "../../../../crates/synapse-owned-decode/owned-decode-routing/q8ingest.rs"]
pub mod q8ingest;
#[path = "../../../../crates/synapse-owned-decode/owned-decode-routing/tensor_source.rs"]
mod tensor_source;

mod owned_decode_routing {
    pub(crate) use crate::tensor_source;
    pub use crate::{decode_error as error, decode_family as family};
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    ensure!(
        args.len() == 4,
        "usage: q8-derive MODEL_SLUG CHECKPOINT_DIRECTORY OUTPUT_CACHE"
    );
    let manifest: serde_json::Value = serde_json::from_str(include_str!("../../models.json"))?;
    let model = &manifest["models"][&args[1]];
    let source = Path::new(&args[2]);
    // Check every ASR source file against the complete file map in models.json.
    if args[1] == "qwen3-asr-1.7b" {
        for (name, digest) in model["files"].as_object().context("unknown model")? {
            ensure!(
                q8artifact::sha256_file(&source.join(name))?
                    == digest.as_str().context("file digest")?,
                "source digest mismatch: {name}"
            );
        }
    }
    let family = match args[1].as_str() {
        "qwen3-asr-1.7b" => q8artifact::QuantizerFamily::Qwen3Asr,
        "qwen3-0.6b" | "qwen3-embedding-0.6b" => q8artifact::QuantizerFamily::Qwen3_0_6b,
        "lfm2-1.2b" => q8artifact::QuantizerFamily::Lfm2_1_2b,
        _ => anyhow::bail!("unsupported quantizer model"),
    };
    let start = Instant::now();
    let source_digest = if family == q8artifact::QuantizerFamily::Qwen3Asr {
        model["checkpoint_digest"]
            .as_str()
            .context("checkpoint digest")?
            .to_string()
    } else {
        q8artifact::sha256_file(&source.join("model.safetensors"))?
    };
    let q8 = q8artifact::derive_q8(
        source,
        Path::new(&args[3]),
        family,
        &source_digest,
        q8asr::QUANTIZER_REVISION,
    )?;
    ensure!(!q8.reused, "measurement requires an empty output cache");
    let f16 = if family == q8artifact::QuantizerFamily::Qwen3Asr {
        Some(q8asr::derive_f16(source, &q8)?)
    } else {
        None
    };
    let elapsed = start.elapsed().as_secs_f64();
    let expected = match family {
        q8artifact::QuantizerFamily::Qwen3_0_6b if args[1] == "qwen3-embedding-0.6b" => {
            Some("17d2fbfeff90269190287f324ed93bab3bb1b4fa4aad98c3fbba1868c01cb0f2")
        }
        q8artifact::QuantizerFamily::Lfm2_1_2b => {
            Some("5874faabdce2567dcc0e7339e9547d79421ba312c71e3442c9cc3c4ed3cb47d0")
        }
        _ => None,
    };
    if let Some(expected) = expected {
        ensure!(
            q8.derived_digest == expected,
            "existing-family Q8 digest changed: {}",
            q8.derived_digest
        );
    }
    println!(
        "{}",
        json!({
        "model": args[1], "source_digest": source_digest,
            "quantizer_revision": q8asr::QUANTIZER_REVISION,
            "q8_0_digest": q8.derived_digest,
            "q8_bytes": std::fs::metadata(&q8.object_path)?.len(),
            "f16_digest": f16.as_ref().map(|p| q8artifact::sha256_file(p)).transpose()?,
            "f16_bytes": f16.as_ref().map(std::fs::metadata).transpose()?.map(|m| m.len()),
            "derivation_wall_seconds": elapsed,
        })
    );
    Ok(())
}
