use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::{
    decoders::byte_level::ByteLevel, models::bpe::BPE, normalizers::unicode::NFC,
    pre_tokenizers::PreTokenizerWrapper, AddedToken,
};

use crate::{Error, Result};

use super::language_name;

pub const VOCAB_SHA256: &str = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910";
pub const MERGES_SHA256: &str = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5";
pub const PRETOKENIZER_SHA256: &str =
    "834cffbaa55e22979cb9bee5d9707a882999771feb2eaaf60895ccdc9dca2776";
pub const TOKENIZER_CONFIG_SHA256: &str =
    "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c";
pub const AUDIO_START: u32 = 151669;
pub const AUDIO_END: u32 = 151670;
pub const AUDIO_PAD: u32 = 151676;
pub const ASR_TEXT: u32 = 151704;

/// Template special tokens are inserted as ids, never recognized in caller text.
/// Non-special added tokens retain the pinned processor's ordinary-text behavior.
pub struct Tokenizer {
    ordinary: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn pinned() -> &'static Self {
        static PINNED: OnceLock<Tokenizer> = OnceLock::new();
        PINNED.get_or_init(|| {
            Self::from_pinned_bytes(
                include_bytes!("../../assets/qwen3-asr/vocab.json"),
                include_bytes!("../../assets/qwen3-asr/merges.txt"),
            )
            .expect("compiled Qwen tokenizer seals")
        })
    }

    pub fn from_pinned_bytes(vocab: &[u8], merges: &[u8]) -> Result<Self> {
        let pretokenizer = include_bytes!("../../assets/qwen3-asr/pretokenizer.json");
        let config = include_bytes!("../../assets/qwen3-asr/tokenizer_config.json");
        for (bytes, expected) in [
            (vocab, VOCAB_SHA256),
            (merges, MERGES_SHA256),
            (pretokenizer.as_slice(), PRETOKENIZER_SHA256),
            (config.as_slice(), TOKENIZER_CONFIG_SHA256),
        ] {
            if format!("{:x}", Sha256::digest(bytes)) != expected {
                return Err(Error::InvalidTokenizer(
                    "Qwen tokenizer sha256 differs from pin".into(),
                ));
            }
        }
        let vocab: serde_json::Value =
            serde_json::from_slice(vocab).map_err(|e| Error::InvalidTokenizer(e.to_string()))?;
        let merges = std::str::from_utf8(merges)
            .map_err(|e| Error::InvalidTokenizer(e.to_string()))?
            .lines()
            .skip(1)
            .filter(|line| !line.is_empty())
            .map(|line| {
                let (a, b) = line
                    .split_once(' ')
                    .ok_or_else(|| Error::InvalidTokenizer("invalid merge".into()))?;
                Ok((a.to_owned(), b.to_owned()))
            })
            .collect::<Result<Vec<_>>>()?;
        let model_json = serde_json::to_vec(&serde_json::json!({
            "type": "BPE", "vocab": vocab, "merges": merges,
        }))
        .map_err(|e| Error::InvalidTokenizer(e.to_string()))?;
        let model: BPE = serde_json::from_slice(&model_json)
            .map_err(|e| Error::InvalidTokenizer(e.to_string()))?;
        // qwen-asr 0.0.6 loads AutoProcessor with fix_mistral_regex=True.
        // Its patched, case-aware split differs from plain Qwen2 (e.g. YouTube).
        let pretokenizer: PreTokenizerWrapper = serde_json::from_slice(pretokenizer)
            .map_err(|e| Error::InvalidTokenizer(e.to_string()))?;
        let mut ordinary = tokenizers::Tokenizer::new(model);
        ordinary.with_decoder(Some(ByteLevel::default()));
        ordinary.with_normalizer(Some(NFC));
        ordinary.with_pre_tokenizer(Some(pretokenizer));
        let config: serde_json::Value =
            serde_json::from_slice(config).map_err(|e| Error::InvalidTokenizer(e.to_string()))?;
        let mut added = config["added_tokens_decoder"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, token)| {
                Ok((
                    id.parse::<u32>()
                        .map_err(|e| Error::InvalidTokenizer(e.to_string()))?,
                    serde_json::from_value::<AddedToken>(token.clone())
                        .map_err(|e| Error::InvalidTokenizer(e.to_string()))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        added.sort_by_key(|(id, _)| *id);
        for (id, token) in added {
            ordinary.add_tokens(std::slice::from_ref(&token));
            if ordinary.token_to_id(&token.content) != Some(id) {
                return Err(Error::InvalidTokenizer(
                    "added-token id differs from pin".into(),
                ));
            }
        }
        // Mirrors split_special_tokens=True. Non-special entries such as
        // <asr_text> still encode as added tokens, but cannot alter chat framing.
        ordinary.set_encode_special_tokens(true);
        Ok(Self { ordinary })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.ordinary
            .encode(text, false)
            .map(|encoded| encoded.get_ids().to_vec())
            .map_err(|e| Error::InvalidTokenizer(e.to_string()))
    }

    pub(super) fn decode_generated(&self, ids: &[u32]) -> Result<String> {
        if ids
            .iter()
            .any(|&id| self.ordinary.id_to_token(id).is_none())
        {
            return Err(Error::InvalidAsrOutput("unknown generated token"));
        }
        self.ordinary
            .decode(ids, true)
            .map_err(|_| Error::InvalidAsrOutput("generated token decoding failed"))
    }

    /// Count the exact forwarded context, not a sum of independently encoded terms.
    pub fn context_tokens(&self, terms: &[impl AsRef<str>], prompt: Option<&str>) -> Result<usize> {
        Ok(self.encode(&context(terms, prompt))?.len())
    }

    /// Keep terms in input order, dropping the caller prompt before any trailing term.
    /// The limit applies to context only, excluding template, hint and audio ids.
    pub fn render(&self, request: PromptRequest<'_>) -> Result<Prompt> {
        let hint = request.language.map(language_name).transpose()?;
        let mut used = request.terms.len();
        let mut prompt = request.prompt.filter(|text| !text.is_empty());
        let mut text = context(request.terms, prompt);
        let mut context_ids = self.encode(&text)?;
        if context_ids.len() > request.bias_limit && prompt.is_some() {
            prompt = None;
            text = context(request.terms, prompt);
            context_ids = self.encode(&text)?;
        }
        while context_ids.len() > request.bias_limit && used > 0 {
            used -= 1;
            text = context(&request.terms[..used], prompt);
            context_ids = self.encode(&text)?;
        }
        let context_tokens = context_ids.len();
        let mut ids = vec![151644, 8948, 198]; // <|im_start|>system\n
        ids.extend(context_ids);
        ids.extend([151645, 198, 151644, 872, 198, AUDIO_START]);
        let span = PlaceholderSpan {
            start: ids.len(),
            len: request.audio_tokens,
        };
        ids.resize(
            ids.len()
                .checked_add(span.len)
                .ok_or(Error::InvalidAudio("audio placeholder length overflow"))?,
            AUDIO_PAD,
        );
        ids.extend([AUDIO_END, 151645, 198, 151644, 77091, 198]);
        if let Some(name) = hint {
            ids.extend(self.encode(&format!("language {name}"))?);
            ids.push(ASR_TEXT);
        }
        Ok(Prompt {
            ids,
            span,
            context: text,
            context_tokens,
            bias_applied: BiasApplied {
                terms_used: used,
                terms_dropped: request.terms.len() - used,
                prompt_used: prompt.is_some(),
            },
        })
    }
}

fn context(terms: &[impl AsRef<str>], prompt: Option<&str>) -> String {
    let mut text = terms
        .iter()
        .map(AsRef::as_ref)
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(prompt) = prompt.filter(|text| !text.is_empty()) {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(prompt);
    }
    text
}

pub struct PromptRequest<'a> {
    pub terms: &'a [&'a str],
    pub prompt: Option<&'a str>,
    /// Language code selecting the hint; region subtags use their base language.
    pub language: Option<&'a str>,
    pub audio_tokens: usize,
    pub bias_limit: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BiasApplied {
    pub terms_used: usize,
    pub terms_dropped: usize,
    pub prompt_used: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaceholderSpan {
    pub start: usize,
    pub len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    pub ids: Vec<u32>,
    pub span: PlaceholderSpan,
    pub context: String,
    pub context_tokens: usize,
    pub bias_applied: BiasApplied,
}
