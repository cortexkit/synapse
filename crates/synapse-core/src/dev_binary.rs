//! Executable aliases for development runs, kept separate from installed images.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// Publish a content-addressed executable copy beside the built binary under a
/// development name. The scratch argument is retained for callers but is no
/// longer used; calls with identical bytes reuse one published copy.
pub fn ckdev_binary(binary: impl AsRef<Path>, _scratch: impl AsRef<Path>) -> io::Result<PathBuf> {
    content_addressed_copy(binary.as_ref())
}

/// Certification must execute the same inode that was hashed, not a copy.
/// Refuse a cross-volume layout rather than weakening that attestation.
pub fn ckdev_binary_hard_link(
    binary: impl AsRef<Path>,
    scratch: impl AsRef<Path>,
) -> io::Result<PathBuf> {
    hard_link_alias(binary.as_ref(), scratch.as_ref())
}

fn development_name(binary: &Path) -> io::Result<&str> {
    let name = binary
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "executable needs a UTF-8 file name",
            )
        })?;
    Ok(name.strip_prefix("ck-").unwrap_or(name))
}

fn content_addressed_copy(binary: &Path) -> io::Result<PathBuf> {
    let name = development_name(binary)?;
    let mut source = File::open(binary)?;
    let source_size = source.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hex::encode(hasher.finalize());
    let parent = binary.parent().unwrap_or_else(|| Path::new("."));
    let dir = parent.join("ckdev-exec").join(&digest[..16]);
    fs::create_dir_all(&dir)?;
    let destination = dir.join(format!("ckdev-{name}"));

    if destination_exists_with_size(&destination, source_size)? {
        return Ok(destination);
    }

    let (mut temporary, temporary_path) = create_temporary_file(&dir, name)?;
    let publication = (|| {
        let mut source = File::open(binary)?;
        let mut copied_hasher = Sha256::new();
        let mut copied_size = 0_u64;
        loop {
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            temporary.write_all(&buffer[..read])?;
            copied_hasher.update(&buffer[..read]);
            copied_size += read as u64;
        }
        if copied_size != source_size || hex::encode(copied_hasher.finalize()) != digest {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "executable changed while its development copy was being made",
            ));
        }
        temporary.flush()?;
        set_executable_permissions(binary, &temporary_path)?;
        drop(temporary);

        if destination_exists_with_size(&destination, source_size)? {
            fs::remove_file(&temporary_path)?;
            return Ok(());
        }
        match fs::rename(&temporary_path, &destination) {
            Ok(()) => Ok(()),
            Err(rename_error) => {
                if destination_exists_with_size(&destination, source_size)? {
                    fs::remove_file(&temporary_path)?;
                    Ok(())
                } else {
                    Err(rename_error)
                }
            }
        }
    })();

    if publication.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    publication?;
    Ok(destination)
}

fn destination_exists_with_size(destination: &Path, expected_size: u64) -> io::Result<bool> {
    match fs::metadata(destination) {
        Ok(metadata) if metadata.is_file() && metadata.len() == expected_size => Ok(true),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "existing development executable copy has a different size",
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn create_temporary_file(dir: &Path, name: &str) -> io::Result<(File, PathBuf)> {
    loop {
        let temporary_path = dir.join(format!(
            ".ckdev-{name}-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((file, temporary_path)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

#[cfg(unix)]
fn set_executable_permissions(binary: &Path, copy: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(binary)?.permissions();
    permissions.set_mode(permissions.mode() | 0o111);
    fs::set_permissions(copy, permissions)
}

#[cfg(not(unix))]
fn set_executable_permissions(_binary: &Path, _copy: &Path) -> io::Result<()> {
    Ok(())
}

fn hard_link_alias(binary: &Path, scratch: &Path) -> io::Result<PathBuf> {
    let name = development_name(binary)?;
    let dir = scratch.join(format!(
        "dev-bin-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir)?;
    let destination = dir.join(format!("ckdev-{name}"));
    fs::hard_link(binary, &destination)?;
    Ok(destination)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_preserve_bytes_names_and_extensions() {
        let root = test_root("names");
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
                use std::os::unix::fs::PermissionsExt;
                assert_ne!(
                    fs::metadata(&alias).unwrap().permissions().mode() & 0o111,
                    0
                );
            }
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repeated_aliases_reuse_one_copy() {
        let root = test_root("reuse");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("ck-synapse");
        fs::write(&source, b"one build image").unwrap();

        let first = ckdev_binary(&source, &root).unwrap();
        let second = ckdev_binary(&source, &root).unwrap();

        assert_eq!(first, second);
        assert_eq!(fs::read(&first).unwrap(), b"one build image");
        assert_eq!(fs::read_dir(first.parent().unwrap()).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn different_bytes_use_different_content_directories() {
        let root = test_root("different");
        fs::create_dir_all(&root).unwrap();
        let first_source = root.join("ck-synapse");
        let second_source = root.join("ck-worker");
        fs::write(&first_source, b"first image").unwrap();
        fs::write(&second_source, b"second image").unwrap();

        let first = ckdev_binary(&first_source, &root).unwrap();
        let second = ckdev_binary(&second_source, &root).unwrap();

        assert_ne!(first.parent(), second.parent());
        assert_eq!(fs::read(first).unwrap(), b"first image");
        assert_eq!(fs::read(second).unwrap(), b"second image");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_aliases_publish_one_complete_copy() {
        let root = test_root("concurrent");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("ck-synapse");
        let bytes = vec![0x5a; 32 * 1024 * 1024];
        fs::write(&source, &bytes).unwrap();

        let digest = hex::encode(Sha256::digest(&bytes));
        let destination = root
            .join("ckdev-exec")
            .join(&digest[..16])
            .join("ckdev-synapse");
        let expected_size = bytes.len() as u64;
        let finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let partial_seen = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observer = {
            let destination = destination.clone();
            let finished = std::sync::Arc::clone(&finished);
            let partial_seen = std::sync::Arc::clone(&partial_seen);
            std::thread::spawn(move || {
                while !finished.load(Ordering::Acquire) {
                    if fs::metadata(&destination)
                        .is_ok_and(|metadata| metadata.len() != expected_size)
                    {
                        partial_seen.store(true, Ordering::Release);
                        break;
                    }
                    std::thread::yield_now();
                }
            })
        };

        let start = std::sync::Arc::new(std::sync::Barrier::new(8));
        let threads = (0..8)
            .map(|_| {
                let source = source.clone();
                let root = root.clone();
                let start = std::sync::Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    ckdev_binary(source, root)
                })
            })
            .collect::<Vec<_>>();
        let results = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        finished.store(true, Ordering::Release);
        observer.join().unwrap();

        assert!(
            !partial_seen.load(Ordering::Acquire),
            "final executable path must never expose a partial copy"
        );
        let aliases = results.into_iter().map(Result::unwrap).collect::<Vec<_>>();

        assert!(aliases.iter().all(|alias| alias == &aliases[0]));
        assert_eq!(fs::read(&aliases[0]).unwrap(), bytes);
        assert_eq!(
            fs::read_dir(aliases[0].parent().unwrap()).unwrap().count(),
            1
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn certification_alias_still_uses_the_built_inode() {
        use std::os::unix::fs::MetadataExt;

        let root = test_root("certification");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("ck-synapse");
        fs::write(&source, b"built image").unwrap();

        let alias = ckdev_binary_hard_link(&source, &root).unwrap();

        assert_eq!(
            fs::metadata(&source).unwrap().ino(),
            fs::metadata(alias).unwrap().ino()
        );
        fs::remove_dir_all(root).unwrap();
    }

    fn test_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "synapse-alias-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
}
