use serde::{Deserialize, Serialize};

use crate::{Error, Result};

use super::{language_code, language_name, Tokenizer};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedTranscript {
    pub text: String,
    /// Contract language code, not the model's English language name.
    pub language: Option<String>,
}

impl Tokenizer {
    /// Parse only engine-generated ids, never prompt ids. A language hint has
    /// already supplied the language prefix and marker in the prompt, so its
    /// continuation is transcript text. The raw model output is not exposed.
    /// A regional hint is reported as its canonical base code from the table.
    pub fn parse(&self, generated_ids: &[u32], hint: Option<&str>) -> Result<ParsedTranscript> {
        let language = hint.map(language_name).transpose()?;
        let decoded = self.decode_generated(generated_ids)?;
        if let Some(name) = language {
            return Ok(ParsedTranscript {
                text: decoded.trim().to_owned(),
                language: language_code(name).map(str::to_owned),
            });
        }
        let (prefix, text) = decoded
            .split_once("<asr_text>")
            .ok_or(Error::InvalidAsrOutput("missing <asr_text> marker"))?;
        let name = prefix
            .trim()
            .strip_prefix("language ")
            .filter(|name| !name.is_empty() && !name.contains('\n'))
            .ok_or(Error::InvalidAsrOutput("missing language line"))?;
        Ok(ParsedTranscript {
            text: text.trim().to_owned(),
            language: language_code(name).map(str::to_owned),
        })
    }
}
