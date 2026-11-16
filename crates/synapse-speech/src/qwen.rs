//! Qwen3-ASR's pinned Whisper-style frontend. Prompt encoding is opt-in so
//! Parakeet-only users do not acquire the BPE encoder or its regex dependencies.

mod frontend;
mod language;
#[cfg(feature = "qwen")]
mod parse;
#[cfg(feature = "qwen")]
mod prompt;

pub use crate::timing::align_words;
pub use frontend::{audio_tokens, mel_frames, Features, Frontend};
pub use language::{language_code, language_name, LANGUAGES};
#[cfg(feature = "qwen")]
pub use parse::ParsedTranscript;
#[cfg(feature = "qwen")]
pub use prompt::*;

pub const FRONTEND_REVISION: &str = "synapse-qwen3-asr-whisper-frontend-v1";
pub const FFT_LENGTH: usize = 400;
pub const MEL_BINS: usize = 128;
pub const HOP_LENGTH: usize = 160;
