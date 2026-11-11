use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, Result, Triple, BLANK, PAD_TOKEN, VOCAB_SIZE};

pub const TOKENIZER_SHA256: &str =
    "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91";
const TOKENIZER: &[u8] = include_bytes!("../assets/tokenizer.json");

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub word: String,
    pub start_ms: f64,
    pub end_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    pub words: Vec<Word>,
    pub no_speech: bool,
}

/// Decode-only view of the pinned BPE vocabulary. The pinned decoder is solely
/// Metaspace (no byte fallback, cleanup, or extra punctuation rules), so encoding
/// and native regular-expression libraries are unnecessary on mobile targets.
pub struct Tokenizer {
    pieces: Vec<String>,
}

impl Tokenizer {
    pub fn pinned() -> &'static Self {
        static PINNED: OnceLock<Tokenizer> = OnceLock::new();
        PINNED.get_or_init(|| Self::from_pinned_bytes(TOKENIZER).expect("compiled tokenizer seal"))
    }

    /// Accept only the pinned tokenizer, not an arbitrary BPE with a different
    /// decoding pipeline. This also lets model loaders validate downloaded bytes.
    pub fn from_pinned_bytes(bytes: &[u8]) -> Result<Self> {
        if format!("{:x}", Sha256::digest(bytes)) != TOKENIZER_SHA256 {
            return Err(Error::InvalidTokenizer(
                "sha256 differs from pinned tokenizer".into(),
            ));
        }
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| Error::InvalidTokenizer(e.to_string()))?;
        let vocab = value["model"]["vocab"]
            .as_object()
            .ok_or_else(|| Error::InvalidTokenizer("missing vocabulary".into()))?;
        let mut pieces = vec![String::new(); VOCAB_SIZE];
        for (piece, id) in vocab {
            if let Some(id) = id.as_u64().filter(|&id| id < VOCAB_SIZE as u64) {
                pieces[id as usize] = piece.clone();
            }
        }
        for added in value["added_tokens"].as_array().unwrap() {
            let id = added["id"].as_u64().unwrap() as usize;
            if id < VOCAB_SIZE {
                pieces[id] = added["content"].as_str().unwrap().to_owned();
            }
        }
        Ok(Self { pieces })
    }

    pub fn piece(&self, id: u32) -> Result<&str> {
        self.pieces
            .get(id as usize)
            .map(String::as_str)
            .ok_or_else(|| Error::InvalidTokenizer(format!("unknown token {id}")))
    }

    /// Matches ParakeetTokenizer.decode with its defaults: collapse adjacent
    /// identical ids, then remove pad ids, then Metaspace decode without cleanup.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut text = String::new();
        let mut previous = None;
        let mut decoded = 0;
        for &id in ids {
            if previous.replace(id) == Some(id) || id == PAD_TOKEN {
                continue;
            }
            let piece = self.piece(id)?;
            let piece = if decoded == 0 {
                piece.strip_prefix('▁').unwrap_or(piece)
            } else {
                piece
            };
            text.push_str(&piece.replace('▁', " "));
            decoded += 1;
        }
        Ok(text)
    }

    /// Timing uses the original input duration, including fractional milliseconds,
    /// rather than the possibly longer resampled clip or rounded admission time.
    pub fn transcript(&self, triples: &[Triple], audio_duration_ms: f64) -> Result<Transcript> {
        if !audio_duration_ms.is_finite() || audio_duration_ms < 0.0 {
            return Err(Error::InvalidAudio("invalid duration"));
        }
        let mut groups: Vec<Vec<Triple>> = Vec::new();
        for &triple in triples {
            if triple[0] == BLANK || triple[2] > 4 {
                return Err(Error::InvalidJoint("invalid emitted triple"));
            }
            if groups.is_empty() || self.piece(triple[0])?.starts_with('▁') {
                groups.push(Vec::new());
            }
            groups.last_mut().unwrap().push(triple);
        }
        let mut words = Vec::with_capacity(groups.len());
        for group in groups {
            let first = group[0];
            let last = group[group.len() - 1];
            words.push(Word {
                word: self
                    .decode(&group.iter().map(|x| x[0]).collect::<Vec<_>>())?
                    .trim()
                    .to_owned(),
                start_ms: (80.0 * f64::from(first[1])).min(audio_duration_ms),
                end_ms: (80.0 * (f64::from(last[1]) + f64::from(last[2].max(1))))
                    .min(audio_duration_ms),
            });
        }
        Ok(Transcript {
            text: self.decode(&triples.iter().map(|x| x[0]).collect::<Vec<_>>())?,
            words,
            no_speech: triples.is_empty(),
        })
    }
}
