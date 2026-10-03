//! Model manifest for the owned embedding and reranking lanes.
//!
//! `bench/parity/models.json` pins, per model, the checkpoint revision and
//! digests, the architecture parameters each engine needs, the input grammar
//! and the rerank head, and per backend profile the numeric choices that enter
//! a lane's fingerprint. This crate is the only code that reads or writes that
//! file: it validates it without any download (`validate`), checks it against
//! the pinned checkpoints (`checkpoint`), produces the converted packages the
//! worker profiles load (`convert`) and computes the Vulkan memory floors
//! (`vulkan`).

pub mod arch;
pub mod canonical;
#[cfg(feature = "checkpoints")]
pub mod checkpoint;
pub mod convert;
pub mod hadamard;
pub mod inventory;
pub mod manifest;
pub mod oracle;
pub mod preload;
pub mod rules;
pub mod safetensors;
pub mod validate;
pub mod vulkan;

use std::path::{Path, PathBuf};

/// Directory holding `models.json` and the files it references, resolved from
/// this crate's own location so tests and tools agree on it.
pub fn parity_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// Repository root, two levels above `bench/parity/`.
pub fn repo_root() -> PathBuf {
    parity_dir()
        .parent()
        .and_then(Path::parent)
        .expect("bench/parity sits two levels below the repository root")
        .to_path_buf()
}

/// Error type shared by every check in this crate. The message names the
/// model, profile or file at fault, so a failing gate is actionable as printed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParityError(pub String);

impl std::fmt::Display for ParityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParityError {}

pub type Result<T> = std::result::Result<T, ParityError>;

/// Build a `ParityError` with `format!` syntax.
#[macro_export]
macro_rules! perr {
    ($($arg:tt)*) => { $crate::ParityError(format!($($arg)*)) };
}
