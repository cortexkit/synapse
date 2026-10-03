//! Checks against the downloaded pinned checkpoints.
//!
//! These complement `validate` with what the committed copies cannot show:
//! that the real files have the pinned digests, that the real tensor index is
//! the committed one, and that the pinned tokenizer encodes as the grammar
//! says (every pinned token id, single-id `yes` and `no`, special and
//! terminal tokens in place).

use std::path::{Path, PathBuf};

use serde_json::Value;
use tokenizers::Tokenizer;

use crate::arch::check_against_config;
use crate::canonical::{sha256_file, sha256_hex};
use crate::manifest::{Family, Grammar, Manifest, Model, Operation, TokenRef};
use crate::safetensors::read_header_bytes;
use crate::{perr, Result};

/// Snapshot directory of a pinned revision inside a Hugging Face hub cache.
pub fn snapshot_dir(hf_cache: &Path, model: &Model) -> PathBuf {
    hf_cache
        .join(format!("models--{}", model.hf_repo.replace('/', "--")))
        .join("snapshots")
        .join(&model.hf_revision)
}

/// Run every checkpoint check for one model.
pub fn check_checkpoint(
    manifest: &Manifest,
    slug: &str,
    dir: &Path,
    parity_dir: &Path,
) -> Result<()> {
    let model = manifest.model(slug)?;
    let at = |error: crate::ParityError| perr!("{slug} at {}: {error}", dir.display());
    for (file, pinned) in &model.files {
        let path = dir.join(file);
        let digest = sha256_file(&path).map_err(|error| at(perr!("hash {file}: {error}")))?;
        if &digest != pinned {
            return Err(at(perr!(
                "{file} has SHA-256 {digest}, manifest pins {pinned}"
            )));
        }
    }
    let header = read_header_bytes(&dir.join("model.safetensors")).map_err(at)?;
    if sha256_hex(&header) != model.tensor_index_sha256 {
        return Err(at(perr!(
            "model.safetensors header differs from tensor_index_sha256"
        )));
    }
    let committed = parity_dir
        .join("checkpoints")
        .join(slug)
        .join("tensor-index.json");
    if std::fs::read(&committed)
        .map_err(|error| at(perr!("read {}: {error}", committed.display())))?
        != header
    {
        return Err(at(perr!(
            "committed tensor index differs from the checkpoint header"
        )));
    }
    let config: Value = read_json(&dir.join("config.json")).map_err(at)?;
    let tokenizer_config = if model.files.contains_key("tokenizer_config.json") {
        Some(read_json(&dir.join("tokenizer_config.json")).map_err(at)?)
    } else {
        None
    };
    check_against_config(
        slug,
        &model.architecture,
        &config,
        tokenizer_config.as_ref(),
    )
    .map_err(at)?;

    let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|error| at(perr!("load tokenizer.json: {error}")))?;
    // The gte reranker's tokenizer.json enables fixed-length padding and
    // truncation. The grammar describes one unpadded input; padding is the
    // batcher's job, so both are switched off before probing.
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|error| at(perr!("disable truncation: {error}")))?;
    check_tokenizer(model, &tokenizer).map_err(at)?;

    if let Some(template) = &model.grammar.template {
        let upstream = std::fs::read(dir.join("README.md"))
            .map_err(|error| at(perr!("read README.md: {error}")))?;
        let oracle = std::fs::read(parity_dir.join(&template.oracle.path))
            .map_err(|error| at(perr!("read oracle: {error}")))?;
        if upstream != oracle {
            return Err(at(perr!(
                "committed template oracle differs from the checkpoint's README.md"
            )));
        }
    }
    Ok(())
}

/// At most the first and last eight ids, so an error stays readable.
fn preview(ids: &[u32]) -> String {
    if ids.len() <= 16 {
        format!("{ids:?}")
    } else {
        format!(
            "{:?} … {:?} ({} ids)",
            &ids[..8],
            &ids[ids.len() - 8..],
            ids.len()
        )
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|error| perr!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| perr!("parse {}: {error}", path.display()))
}

fn encode(tokenizer: &Tokenizer, text: &str, special: bool) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, special)
        .map_err(|error| perr!("encode {text:?}: {error}"))?
        .get_ids()
        .to_vec())
}

/// `text` must encode, without special tokens, to exactly one id, and that id
/// must be the one recorded.
pub fn check_single_id(tokenizer: &Tokenizer, token: &TokenRef) -> Result<()> {
    let ids = encode(tokenizer, &token.text, false)?;
    if ids != [token.id] {
        return Err(perr!(
            "qwen_readout_mismatch: `{}` encodes to {ids:?}, expected the single id {}",
            token.text,
            token.id
        ));
    }
    check_token_id(tokenizer, token)
}

fn check_token_id(tokenizer: &Tokenizer, token: &TokenRef) -> Result<()> {
    match tokenizer.token_to_id(&token.text) {
        Some(id) if id == token.id => Ok(()),
        other => Err(perr!(
            "token `{}` has id {other:?} in the pinned tokenizer, manifest records {}",
            token.text,
            token.id
        )),
    }
}

pub fn check_tokenizer(model: &Model, tokenizer: &Tokenizer) -> Result<()> {
    let grammar: &Grammar = &model.grammar;
    for token in grammar
        .special_tokens
        .values()
        .chain(&grammar.terminal_tokens)
        .chain([&grammar.pad])
    {
        check_token_id(tokenizer, token)?;
    }
    for token in [&grammar.readout.yes, &grammar.readout.no]
        .into_iter()
        .flatten()
    {
        check_single_id(tokenizer, token)?;
    }
    let probe = "Synapse parity probe: the quick brown fox.";
    match (model.architecture.family, model.operation) {
        (Family::Modernbert, Operation::Embed) => {
            let ids = encode(tokenizer, probe, true)?;
            let (cls, sep) = (
                grammar.special_tokens["cls"].id,
                grammar.special_tokens["sep"].id,
            );
            if ids.first() != Some(&cls)
                || ids.last() != Some(&sep)
                || ids.iter().filter(|id| **id == sep).count() != 1
            {
                return Err(perr!(
                    "single-sequence encoding {} is not [cls] … [sep]",
                    preview(&ids)
                ));
            }
        }
        (Family::Modernbert, Operation::Rerank) => {
            let ids = tokenizer
                .encode((probe, "a candidate document"), true)
                .map_err(|error| perr!("encode pair: {error}"))?
                .get_ids()
                .to_vec();
            let (cls, sep) = (
                grammar.special_tokens["cls"].id,
                grammar.special_tokens["sep"].id,
            );
            if ids.first() != Some(&cls)
                || ids.last() != Some(&sep)
                || ids.iter().filter(|id| **id == sep).count() != 2
            {
                return Err(perr!(
                    "pair encoding {} is not [cls] q [sep] d [sep]",
                    preview(&ids)
                ));
            }
        }
        (Family::Qwen3, Operation::Embed) => {
            let terminal = grammar.terminal_tokens[0].id;
            let with = encode(tokenizer, probe, true)?;
            let without = encode(tokenizer, probe, false)?;
            if with.last() != Some(&terminal)
                || with.iter().filter(|id| **id == terminal).count() != 1
                || without.contains(&terminal)
            {
                return Err(perr!(
                    "Qwen3 embed encoding {} does not end in exactly one terminal {terminal}",
                    preview(&with)
                ));
            }
        }
        (Family::Qwen3, Operation::Rerank) => {
            let template = grammar
                .template
                .as_ref()
                .ok_or_else(|| perr!("no template"))?;
            let pieces = [
                encode(tokenizer, &template.prefix, false)?,
                encode(tokenizer, &template.suffix, false)?,
            ];
            for token in grammar.special_tokens.values() {
                if !pieces.iter().any(|ids| ids.contains(&token.id)) {
                    return Err(perr!(
                        "template special token `{}` is not a single id in the template",
                        token.text
                    ));
                }
            }
        }
    }
    Ok(())
}
