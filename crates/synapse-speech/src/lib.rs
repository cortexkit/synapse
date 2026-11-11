//! Platform-independent Parakeet TDT speech processing. Engines supply the neural
//! predictor and joint; this crate owns audio, features, decode policy and text.

mod audio;
mod decode;
mod frontend;
mod text;

pub use audio::{decode_wav, resample, Audio, MAX_AUDIO_BYTES, SAMPLE_RATE};
pub use decode::{greedy, DecodeOutput, Joint, JointLogits, TraceStep, Triple};
pub use frontend::{actual_mel_frames, valid_encoder_frames, valid_mel_frames, Features, Frontend};
pub use text::{Tokenizer, Transcript, Word, TOKENIZER_SHA256};

pub const SPEECH_FRONTEND_REVISION: &str = "synapse-parakeet-speech-frontend-v1";
pub const DECODE_RULE_REVISION: &str = "synapse-parakeet-tdt-greedy-v1";
pub const BLANK: u32 = 8192;
pub const PAD_TOKEN: u32 = 2;
pub const VOCAB_SIZE: usize = 8193;
pub const DURATIONS: [u32; 5] = [0, 1, 2, 3, 4];
pub const MAX_SYMBOLS_PER_FRAME: usize = 10;
pub const MEL_BINS: usize = 128;
pub const HOP_LENGTH: usize = 160;
pub const WINDOW_LENGTH: usize = 400;
pub const FFT_LENGTH: usize = 512;
pub const PREEMPHASIS: f32 = 0.97;
pub const LOG_GUARD: f32 = 5.960_464_5e-8;
pub const NORMALIZATION_EPSILON: f32 = 1e-5;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Callers map this error to invalid_params with field "audio".
    #[error("invalid audio: {0}")]
    InvalidAudio(&'static str),
    #[error("invalid joint output: {0}")]
    InvalidJoint(&'static str),
    #[error("invalid tokenizer or token: {0}")]
    InvalidTokenizer(String),
}

pub type Result<T> = std::result::Result<T, Error>;
