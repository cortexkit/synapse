use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{TokenBatch, TokenIds, TruncationDisclosure};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub max_tokens: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self { max_tokens: 512 }
    }
}

#[derive(Debug, Error)]
pub enum TokenizationError {
    #[error("load tokenizer {path}: {message}")]
    Load { path: String, message: String },
    #[error("sanitize tokenizer: {0}")]
    Sanitize(String),
    #[error("encode item {index}: {message}")]
    Encode { index: usize, message: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizedItem {
    pub ids: TokenIds,
    pub disclosure: TruncationDisclosure,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizedBatch {
    pub batch: TokenBatch,
    pub disclosures: Vec<TruncationDisclosure>,
    pub real_token_counts: Vec<u32>,
}

#[derive(Clone)]
pub struct SanitizedTokenizer {
    tokenizer: Tokenizer,
    max_tokens: usize,
    sanitized_sha256: String,
}

impl SanitizedTokenizer {
    pub fn from_file(
        path: impl AsRef<Path>,
        config: TokenizerConfig,
    ) -> Result<Self, TokenizationError> {
        let path = path.as_ref();
        let mut tokenizer =
            Tokenizer::from_file(path).map_err(|error| TokenizationError::Load {
                path: path.display().to_string(),
                message: error.to_string(),
            })?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|error| TokenizationError::Sanitize(error.to_string()))?;
        let sanitized_sha256 = tokenizer_sha256(&tokenizer)?;
        Ok(Self {
            tokenizer,
            max_tokens: config.max_tokens.max(1),
            sanitized_sha256,
        })
    }

    #[must_use]
    pub fn sanitized_sha256(&self) -> &str {
        &self.sanitized_sha256
    }

    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn tokenize(&self, text: &str) -> Result<TokenizedItem, TokenizationError> {
        self.tokenize_one(0, text)
    }

    pub fn tokenize_batch<'text, I>(&self, texts: I) -> Result<TokenizedBatch, TokenizationError>
    where
        I: IntoIterator<Item = &'text str>,
    {
        let items = texts
            .into_iter()
            .enumerate()
            .map(|(index, text)| self.tokenize_one(index, text))
            .collect::<Result<Vec<_>, _>>()?;
        let batch = TokenBatch {
            items: items.iter().map(|item| item.ids.clone()).collect(),
        };
        let disclosures = items
            .iter()
            .map(|item| item.disclosure.clone())
            .collect::<Vec<_>>();
        let real_token_counts = disclosures
            .iter()
            .map(|disclosure| disclosure.effective_tokens)
            .collect();
        Ok(TokenizedBatch {
            batch,
            disclosures,
            real_token_counts,
        })
    }

    fn tokenize_one(&self, index: usize, text: &str) -> Result<TokenizedItem, TokenizationError> {
        let encoding =
            self.tokenizer
                .encode(text, true)
                .map_err(|error| TokenizationError::Encode {
                    index,
                    message: error.to_string(),
                })?;
        let submitted_tokens = encoding.get_ids().len();
        let mut ids = encoding.get_ids().to_vec();
        if ids.len() > self.max_tokens {
            ids.truncate(self.max_tokens);
        }
        let effective_tokens = ids.len();
        Ok(TokenizedItem {
            ids,
            disclosure: TruncationDisclosure {
                submitted_tokens: submitted_tokens.min(u32::MAX as usize) as u32,
                effective_tokens: effective_tokens.min(u32::MAX as usize) as u32,
                truncated: submitted_tokens > effective_tokens,
            },
        })
    }
}

fn tokenizer_sha256(tokenizer: &Tokenizer) -> Result<String, TokenizationError> {
    let serialized = tokenizer
        .to_string(false)
        .map_err(|error| TokenizationError::Sanitize(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(serialized.as_bytes());
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use ahash::AHashMap;

    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::{PaddingParams, PaddingStrategy};

    #[test]
    fn tokenizer_strips_padding_and_reports_our_truncation() {
        let path = std::env::temp_dir().join(format!(
            "synapse-tokenizer-{}-{}.json",
            std::process::id(),
            unique_suffix()
        ));
        write_wordlevel_tokenizer(&path);

        let tokenizer = SanitizedTokenizer::from_file(&path, TokenizerConfig { max_tokens: 3 })
            .expect("load sanitized tokenizer");
        let tokenized = tokenizer
            .tokenize_batch(["a b", "a b c d e"])
            .expect("tokenize texts");
        let _ = std::fs::remove_file(&path);

        assert_eq!(tokenized.real_token_counts, vec![2, 3]);
        assert_eq!(
            tokenized.batch.items[0].len(),
            2,
            "saved padding must not leak into ids"
        );
        assert_eq!(
            tokenized.disclosures[1],
            TruncationDisclosure {
                submitted_tokens: 5,
                effective_tokens: 3,
                truncated: true,
            }
        );
    }

    fn write_wordlevel_tokenizer(path: &Path) {
        let mut vocab = AHashMap::new();
        for (index, token) in ["[UNK]", "a", "b", "c", "d", "e"].iter().enumerate() {
            vocab.insert((*token).to_string(), index as u32);
        }
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("build wordlevel tokenizer");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(128),
            ..Default::default()
        }));
        tokenizer.save(path, false).expect("save tokenizer fixture");
    }

    fn unique_suffix() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time is after unix epoch")
            .as_nanos()
    }
}
