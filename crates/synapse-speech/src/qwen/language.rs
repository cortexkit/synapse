use crate::{Error, Result};

/// All 30 names from the pinned Qwen3-ASR config. The served catalog declaration
/// narrows these languages to those supported by its Parakeet word-timing companion.
/// Codes use ISO 639-1 when available, otherwise ISO 639-3.
pub const LANGUAGES: &[(&str, &str)] = &[
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("fil", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
];

/// Language codes with a region subtag match their base language.
pub fn language_name(code: &str) -> Result<&'static str> {
    let base = code.split('-').next().unwrap_or(code);
    LANGUAGES
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(base))
        .map(|(_, name)| *name)
        .ok_or_else(|| Error::UnsupportedLanguage(code.to_owned()))
}

/// Unknown model names, including `None`, report no detected language.
pub fn language_code(name: &str) -> Option<&'static str> {
    LANGUAGES
        .iter()
        .find(|(_, candidate)| *candidate == name)
        .map(|(code, _)| *code)
}
