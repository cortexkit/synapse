//! Executable aliases for development runs, kept separate from installed images.

use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// Link a built executable into the caller's scratch tree under a development
/// name. Copying is permitted when the scratch tree is on another volume.
/// Each call has its own directory so concurrent children never replace an
/// executable that is still running (including on Windows).
pub fn ckdev_binary(binary: impl AsRef<Path>, scratch: impl AsRef<Path>) -> io::Result<PathBuf> {
    alias(binary.as_ref(), scratch.as_ref(), true)
}

/// Certification must execute the same inode that was hashed, not a copy.
/// Refuse a cross-volume layout rather than weakening that attestation.
pub fn ckdev_binary_hard_link(
    binary: impl AsRef<Path>,
    scratch: impl AsRef<Path>,
) -> io::Result<PathBuf> {
    alias(binary.as_ref(), scratch.as_ref(), false)
}

fn alias(binary: &Path, scratch: &Path, copy_allowed: bool) -> io::Result<PathBuf> {
    let name = binary
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "executable needs a UTF-8 file name",
            )
        })?;
    let name = name.strip_prefix("ck-").unwrap_or(name);
    let dir = scratch.join(format!(
        "dev-bin-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir)?;
    let destination = dir.join(format!("ckdev-{name}"));
    match fs::hard_link(binary, &destination) {
        Ok(()) => {}
        Err(error) if copy_allowed && is_cross_volume(&error) => {
            fs::copy(binary, &destination)?;
        }
        Err(error) => return Err(error),
    }
    Ok(destination)
}

fn is_cross_volume(error: &io::Error) -> bool {
    // EXDEV on Unix, ERROR_NOT_SAME_DEVICE on Windows.
    error.raw_os_error() == Some(if cfg!(windows) { 17 } else { 18 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_preserve_bytes_names_extensions_and_inode() {
        let root = std::env::temp_dir().join(format!("synapse-alias-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        for name in ["ck-synapse", "ck-synapse-worker-cuda.exe", "mock-worker"] {
            let source = root.join(name);
            fs::write(&source, b"built image").unwrap();
            let alias = ckdev_binary(&source, &root).unwrap();
            assert_eq!(
                alias.file_name().unwrap().to_str().unwrap(),
                format!("ckdev-{}", name.strip_prefix("ck-").unwrap_or(name))
            );
            assert_eq!(fs::read(&alias).unwrap(), b"built image");
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                assert_eq!(
                    fs::metadata(source).unwrap().ino(),
                    fs::metadata(alias).unwrap().ino()
                );
            }
        }
        fs::remove_dir_all(root).unwrap();
    }
}
