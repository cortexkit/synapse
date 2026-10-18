//! Executable aliases for development runs, kept separate from installed images.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};

static NEXT: AtomicU64 = AtomicU64::new(0);

#[cfg(target_os = "macos")]
const MACOS_STRIPPED_COPY_FORMAT: &[u8] = b"macos-strip-debug-map-adhoc-sign/1\0";

/// Publish a content-addressed executable copy under a development name. Unix
/// copies live in a safe temporary directory; Windows keeps its existing layout.
/// The scratch argument is retained for callers but is no longer used; calls
/// with identical bytes reuse one published copy.
pub fn ckdev_binary(binary: impl AsRef<Path>, _scratch: impl AsRef<Path>) -> io::Result<PathBuf> {
    content_addressed_copy(binary.as_ref())
}

/// Certification must execute the same inode that was hashed, not a copy.
/// Refuse a cross-volume layout rather than weakening that attestation.
/// Hard links now reuse a content-addressed directory; certification still gets
/// a hard link to the exact candidate bytes, so attestation is unchanged.
pub fn ckdev_binary_hard_link(
    binary: impl AsRef<Path>,
    scratch: impl AsRef<Path>,
) -> io::Result<PathBuf> {
    hard_link_alias(binary.as_ref(), scratch.as_ref())
}

/// Publish a script to a stable content-addressed path under a development name.
/// The script bytes are written once per unique content and published via `ckdev_binary`.
/// Subsequent calls with identical bytes return the existing executable without rewriting.
pub fn ckdev_script(name: impl AsRef<Path>, bytes: &[u8]) -> io::Result<PathBuf> {
    let name_ref = name.as_ref();
    let file_name = name_ref
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "script needs a UTF-8 file name",
            )
        })?;
    let digest = hex::encode(Sha256::digest(bytes));
    let dir = test_script_dir().join(&digest[..16]);
    fs::create_dir_all(&dir)?;
    let source = dir.join(file_name);
    if !source.is_file() {
        let temporary = dir.join(format!(
            ".{file_name}-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&temporary, bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&temporary, fs::Permissions::from_mode(0o700))?;
        }
        match fs::rename(&temporary, &source) {
            Ok(()) => {}
            Err(_) if source.is_file() => {
                let _ = fs::remove_file(&temporary);
            }
            Err(e) => return Err(e),
        }
    }
    ckdev_binary(&source, &dir)
}

/// Directory used to publish stable development test scripts across test runs.
pub fn test_script_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("SYNAPSE_DEV_SCRIPTS_DIR") {
        return PathBuf::from(path);
    }
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    let mut dir = exe.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    if dir.file_name().and_then(|name| name.to_str()) == Some("deps") {
        dir.pop();
    }
    dir.join("ckdev-scripts")
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
    #[cfg(unix)]
    let base = current_development_binary_base();
    #[cfg(not(unix))]
    let base = binary
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    content_addressed_copy_at(binary, &base)
}

fn content_addressed_copy_at(binary: &Path, base: &Path) -> io::Result<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        content_addressed_copy_at_with_processor(binary, base, Some(&strip_and_sign_macos))
    }
    #[cfg(not(target_os = "macos"))]
    {
        content_addressed_copy_at_with_processor(binary, base, None)
    }
}

/// Post-processes a staged Mach-O copy (strip and re-sign) before it is published.
type CopyProcessor<'a> = &'a dyn Fn(&Path) -> io::Result<()>;

fn content_addressed_copy_at_with_processor(
    binary: &Path,
    base: &Path,
    macos_processor: Option<CopyProcessor<'_>>,
) -> io::Result<PathBuf> {
    let name = development_name(binary)?;
    #[cfg(target_os = "macos")]
    let is_macho = is_macho_binary(binary)?;
    #[cfg(not(target_os = "macos"))]
    let is_macho = false;

    let mut source = File::open(binary)?;
    let source_size = source.metadata()?.len();
    let mut hasher = Sha256::new();
    let mut source_hasher = Sha256::new();
    #[cfg(target_os = "macos")]
    if is_macho {
        hasher.update(MACOS_STRIPPED_COPY_FORMAT);
    }
    let mut buffer = [0; 64 * 1024];
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        source_hasher.update(&buffer[..read]);
    }
    let source_digest = hex::encode(source_hasher.finalize());
    let digest = hex::encode(hasher.finalize());
    let dir = copy_directory(base, &digest)?;
    let destination = dir.join(format!("ckdev-{name}"));

    let cached_copy_exists = if is_macho {
        destination_exists(&destination)?
    } else {
        destination_exists_with_size(&destination, source_size)?
    };
    if cached_copy_exists {
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
        if copied_size != source_size || hex::encode(copied_hasher.finalize()) != source_digest {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "executable changed while its development copy was being made",
            ));
        }
        temporary.flush()?;
        set_executable_permissions(binary, &temporary_path)?;
        drop(temporary);

        #[cfg(target_os = "macos")]
        if is_macho {
            let processor = macos_processor.ok_or_else(|| {
                io::Error::other("macOS Mach-O publication has no strip-and-sign processor")
            })?;
            processor(&temporary_path)?;
        }
        #[cfg(not(target_os = "macos"))]
        let _ = macos_processor;

        let output_size = fs::metadata(&temporary_path)?.len();
        let already_published = if is_macho {
            destination_exists(&destination)?
        } else {
            destination_exists_with_size(&destination, output_size)?
        };
        if already_published {
            fs::remove_file(&temporary_path)?;
            return Ok(());
        }
        match fs::rename(&temporary_path, &destination) {
            Ok(()) => Ok(()),
            Err(rename_error) => {
                let already_published = if is_macho {
                    destination_exists(&destination)?
                } else {
                    destination_exists_with_size(&destination, output_size)?
                };
                if already_published {
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

#[cfg(target_os = "macos")]
fn is_macho_binary(binary: &Path) -> io::Result<bool> {
    let mut source = File::open(binary)?;
    let mut magic = [0; 4];
    match source.read_exact(&mut magic) {
        Ok(()) => Ok(matches!(
            magic,
            [0xfe, 0xed, 0xfa, 0xce]
                | [0xce, 0xfa, 0xed, 0xfe]
                | [0xfe, 0xed, 0xfa, 0xcf]
                | [0xcf, 0xfa, 0xed, 0xfe]
                | [0xca, 0xfe, 0xba, 0xbe]
                | [0xbe, 0xba, 0xfe, 0xca]
                | [0xca, 0xfe, 0xba, 0xbf]
                | [0xbf, 0xba, 0xfe, 0xca]
        )),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(target_os = "macos")]
fn strip_and_sign_macos(copy: &Path) -> io::Result<()> {
    strip_and_sign_macos_with_tools(
        copy,
        Path::new("/usr/bin/strip"),
        Path::new("/usr/bin/codesign"),
    )
}

#[cfg(target_os = "macos")]
fn strip_and_sign_macos_with_tools(
    copy: &Path,
    strip_tool: &Path,
    codesign_tool: &Path,
) -> io::Result<()> {
    run_macos_tool(strip_tool, &["-S"], copy)?;
    run_macos_tool(codesign_tool, &["-f", "-s", "-"], copy)
}

#[cfg(target_os = "macos")]
fn run_macos_tool(tool: &Path, arguments: &[&str], target: &Path) -> io::Result<()> {
    let command_arguments = arguments.join(" ");
    let output = super::child_process::without_launch_nonce(std::process::Command::new(tool))
        .args(arguments)
        .arg(target)
        .output()
        .map_err(|error| {
            io::Error::other(format!(
                "{} {} {} failed: could not start: {error}; stderr: (none)",
                tool.display(),
                command_arguments,
                target.display()
            ))
        })?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stderr = stderr.trim();
    let stderr = if stderr.is_empty() { "(empty)" } else { stderr };
    Err(io::Error::other(format!(
        "{} {} {} failed: {}; stderr: {stderr}",
        tool.display(),
        command_arguments,
        target.display(),
        output.status
    )))
}

#[cfg(unix)]
fn current_development_binary_base() -> PathBuf {
    let tmpdir = std::env::var_os("TMPDIR").map(PathBuf::from);
    let canonical_tmpdir = tmpdir
        .as_deref()
        .and_then(|path| fs::canonicalize(path).ok());
    development_binary_base(tmpdir.as_deref(), canonical_tmpdir.as_deref())
}

#[cfg(unix)]
fn development_binary_base(tmpdir: Option<&Path>, canonical_tmpdir: Option<&Path>) -> PathBuf {
    fn is_macos_transient_directory(path: &Path) -> bool {
        path.starts_with("/var/folders") || path.starts_with("/private/var/folders")
    }

    match (tmpdir, canonical_tmpdir) {
        (Some(written), Some(canonical))
            if written.is_absolute()
                && !is_macos_transient_directory(written)
                && !is_macos_transient_directory(canonical) =>
        {
            written.to_path_buf()
        }
        _ => PathBuf::from("/tmp"),
    }
}

#[cfg(unix)]
fn copy_directory(base: &Path, digest: &str) -> io::Result<PathBuf> {
    let exec_root = base.join("synapse-exec");
    ensure_private_directory(&exec_root)?;
    let directory = exec_root.join(&digest[..16]);
    ensure_private_directory(&directory)?;
    Ok(directory)
}

#[cfg(not(unix))]
fn copy_directory(base: &Path, digest: &str) -> io::Result<PathBuf> {
    let directory = base.join("ckdev-exec").join(&digest[..16]);
    fs::create_dir_all(&directory)?;
    Ok(directory)
}

#[cfg(unix)]
fn ensure_private_directory(directory: &Path) -> io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};

    match fs::DirBuilder::new().mode(0o700).create(directory) {
        Ok(()) => fs::set_permissions(directory, fs::Permissions::from_mode(0o700)),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let metadata = fs::symlink_metadata(directory)?;
            if !metadata.is_dir()
                || metadata.uid() != effective_user_id()
                || metadata.permissions().mode() & 0o022 != 0
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!(
                        "development executable directory is not private: {}",
                        directory.display()
                    ),
                ));
            }
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
        }
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn effective_user_id() -> u32 {
    rustix::process::geteuid().as_raw()
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

fn destination_exists(destination: &Path) -> io::Result<bool> {
    match fs::metadata(destination) {
        Ok(metadata) if metadata.is_file() => Ok(true),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "existing development executable copy is not a file",
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

fn destination_is_hard_link(destination: &Path, binary: &Path) -> io::Result<bool> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let dest_meta = match fs::metadata(destination) {
            Ok(m) => m,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e),
        };
        let bin_meta = fs::metadata(binary)?;
        Ok(dest_meta.ino() == bin_meta.ino() && dest_meta.dev() == bin_meta.dev())
    }
    #[cfg(not(unix))]
    {
        match fs::metadata(destination) {
            Ok(m) => {
                let bin_meta = fs::metadata(binary)?;
                Ok(m.is_file() && m.len() == bin_meta.len())
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }
}

fn hard_link_alias(binary: &Path, scratch: &Path) -> io::Result<PathBuf> {
    let name = development_name(binary)?;
    let mut source = File::open(binary)?;
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
    let dir = scratch.join(&digest[..16]);
    fs::create_dir_all(&dir)?;
    let destination = dir.join(format!("ckdev-{name}"));
    if destination_is_hard_link(&destination, binary)? {
        return Ok(destination);
    }
    let _ = fs::remove_file(&destination);
    match fs::hard_link(binary, &destination) {
        Ok(()) => Ok(destination),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            if destination_is_hard_link(&destination, binary)? {
                Ok(destination)
            } else {
                Err(e)
            }
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn development_binary_base_uses_only_an_absolute_safe_tmpdir() {
        let ordinary = Path::new("/tmp/synapse-check");
        assert_eq!(development_binary_base(None, None), Path::new("/tmp"));
        assert_eq!(
            development_binary_base(
                Some(Path::new("relative")),
                Some(Path::new("/tmp/relative"))
            ),
            Path::new("/tmp")
        );
        assert_eq!(
            development_binary_base(
                Some(Path::new("/var/folders/task")),
                Some(Path::new("/tmp/canonical-safe-path"))
            ),
            Path::new("/tmp")
        );
        assert_eq!(
            development_binary_base(
                Some(Path::new("/private/var/folders/task")),
                Some(Path::new("/private/var/folders/task"))
            ),
            Path::new("/tmp")
        );
        assert_eq!(
            development_binary_base(Some(ordinary), Some(ordinary)),
            ordinary
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_tmpdir_canonicalizing_into_private_var_folders_uses_tmp() {
        let written_link = Path::new("/tmp/synapse-temp-link");
        let canonical_target = Path::new("/private/var/folders/task");
        assert_eq!(
            development_binary_base(Some(written_link), Some(canonical_target)),
            Path::new("/tmp")
        );
    }

    #[cfg(unix)]
    #[test]
    fn refuses_group_or_world_writable_content_directory() {
        use std::os::unix::fs::PermissionsExt;

        let root = test_root("unsafe-directory");
        let source = root.join("ck-synapse");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"unsafe directory test").unwrap();
        let digest = hex::encode(Sha256::digest(b"unsafe directory test"));
        let directory = root.join("synapse-exec").join(&digest[..16]);
        fs::create_dir_all(&directory).unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o777)).unwrap();

        let error = content_addressed_copy_at(&source, &root).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn aliases_preserve_bytes_names_and_extensions() {
        let root = test_root("names");
        fs::create_dir_all(&root).unwrap();
        for name in ["ck-synapse", "ck-synapse-worker-cuda.exe", "mock-worker"] {
            let source = root.join(name);
            fs::write(&source, b"built image").unwrap();
            let alias = content_addressed_copy_at(&source, &root).unwrap();
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

        let first = content_addressed_copy_at(&source, &root).unwrap();
        let second = content_addressed_copy_at(&source, &root).unwrap();

        assert_eq!(first, second);
        assert_eq!(fs::read(&first).unwrap(), b"one build image");
        assert_eq!(fs::read_dir(first.parent().unwrap()).unwrap().count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            for directory in [
                first.parent().unwrap().parent().unwrap(),
                first.parent().unwrap(),
            ] {
                assert_eq!(
                    fs::metadata(directory).unwrap().permissions().mode() & 0o777,
                    0o700
                );
            }
        }
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

        let first = content_addressed_copy_at(&first_source, &root).unwrap();
        let second = content_addressed_copy_at(&second_source, &root).unwrap();

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
            .join("synapse-exec")
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
                    content_addressed_copy_at(&source, &root)
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

    #[cfg(target_os = "macos")]
    #[test]
    fn published_macho_copy_has_no_stabs_and_keeps_function_symbols() {
        let source = std::env::current_exe().unwrap();
        let copy = ckdev_binary(&source, Path::new("unused-scratch")).unwrap();
        let unversioned_digest = hex::encode(Sha256::digest(fs::read(&source).unwrap()));
        let published_directory = copy
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        assert_ne!(
            published_directory,
            &unversioned_digest[..16],
            "Mach-O publications must not reuse the old unstripped cache key"
        );
        let symbols = nm_symbols(&copy);
        let stab_entries = symbols
            .lines()
            .filter(|line| line.split_whitespace().nth(1) == Some("-"))
            .collect::<Vec<_>>();
        assert!(
            stab_entries.is_empty(),
            "debug map entries remain: {stab_entries:?}"
        );

        let source_symbols = nm_symbols(&source);
        let retained_function = source_symbols
            .lines()
            .filter_map(|line| {
                let fields = line.split_whitespace().collect::<Vec<_>>();
                if fields.len() >= 3 && matches!(fields[1], "T" | "t") {
                    Some(fields[2])
                } else {
                    None
                }
            })
            .find(|symbol| {
                symbols
                    .lines()
                    .any(|line| line.split_whitespace().last() == Some(*symbol))
            });
        assert!(
            retained_function.is_some(),
            "no function symbol survived strip -S"
        );

        let verification = std::process::Command::new("/usr/bin/codesign")
            .arg("-v")
            .arg(&copy)
            .output()
            .unwrap();
        assert!(
            verification.status.success(),
            "codesign -v {} failed: {}",
            copy.display(),
            String::from_utf8_lossy(&verification.stderr)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn failed_strip_names_tool_and_file_and_does_not_publish() {
        use std::os::unix::fs::PermissionsExt;

        let root = test_root("strip-failure");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("ck-strip-failure");
        fs::copy(std::env::current_exe().unwrap(), &source).unwrap();
        let failing_strip = root.join("failing-strip");
        fs::write(
            &failing_strip,
            b"#!/bin/sh\nprintf '%s\\n' 'deliberate strip failure' >&2\nexit 17\n",
        )
        .unwrap();
        fs::set_permissions(&failing_strip, fs::Permissions::from_mode(0o700)).unwrap();
        let processor = |copy: &Path| {
            strip_and_sign_macos_with_tools(copy, &failing_strip, Path::new("/usr/bin/codesign"))
        };

        let error = content_addressed_copy_at_with_processor(&source, &root, Some(&processor))
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("failing-strip"),
            "missing tool name: {error}"
        );
        assert!(
            error.contains("ckdev-strip-failure"),
            "missing file name: {error}"
        );
        assert!(
            error.contains("deliberate strip failure"),
            "missing stderr: {error}"
        );

        let mut hasher = Sha256::new();
        hasher.update(MACOS_STRIPPED_COPY_FORMAT);
        hasher.update(fs::read(&source).unwrap());
        let digest = hex::encode(hasher.finalize());
        let directory = root.join("synapse-exec").join(&digest[..16]);
        assert!(!directory.join("ckdev-strip-failure").exists());
        assert_eq!(fs::read_dir(directory).unwrap().count(), 0);
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
        let candidate_digest = hex::encode(Sha256::digest(fs::read(&source).unwrap()));

        let alias = ckdev_binary_hard_link(&source, &root).unwrap();

        assert_eq!(
            hex::encode(Sha256::digest(fs::read(&source).unwrap())),
            candidate_digest
        );
        assert_eq!(
            hex::encode(Sha256::digest(fs::read(&alias).unwrap())),
            candidate_digest
        );
        assert_eq!(
            fs::metadata(&source).unwrap().ino(),
            fs::metadata(alias).unwrap().ino()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repeated_hard_link_aliases_reuse_one_link() {
        let root = test_root("hard-link-reuse");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("ck-synapse");
        fs::write(&source, b"hard link test image").unwrap();

        let first = ckdev_binary_hard_link(&source, &root).unwrap();
        let second = ckdev_binary_hard_link(&source, &root).unwrap();

        assert_eq!(first, second);
        assert_eq!(fs::read(&first).unwrap(), b"hard link test image");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ckdev_script_publishes_and_reuses_stable_path() {
        let script = b"#!/bin/sh\necho test\n";
        let first = ckdev_script("ck-test-script.sh", script).unwrap();
        let second = ckdev_script("ck-test-script.sh", script).unwrap();
        assert_eq!(first, second);
        assert!(first
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("ckdev-"));
        assert_eq!(fs::read(&first).unwrap(), script);
    }

    #[test]
    fn ckdev_script_publishes_distinct_paths_for_different_contents_under_same_name() {
        let first_bytes = if cfg!(windows) {
            b"@echo off\r\necho first\r\n" as &[u8]
        } else {
            b"#!/bin/sh\necho first\n"
        };
        let second_bytes = if cfg!(windows) {
            b"@echo off\r\necho second\r\n" as &[u8]
        } else {
            b"#!/bin/sh\necho second\n"
        };
        let script_name = if cfg!(windows) {
            "ck-same-name.cmd"
        } else {
            "ck-same-name.sh"
        };
        let first = ckdev_script(script_name, first_bytes).unwrap();
        let second = ckdev_script(script_name, second_bytes).unwrap();
        assert_ne!(first, second);
        assert_eq!(fs::read(&first).unwrap(), first_bytes);
        assert_eq!(fs::read(&second).unwrap(), second_bytes);

        let first_out = std::process::Command::new(&first).output().unwrap();
        assert!(String::from_utf8_lossy(&first_out.stdout).contains("first"));
        let second_out = std::process::Command::new(&second).output().unwrap();
        assert!(String::from_utf8_lossy(&second_out.stdout).contains("second"));
    }

    fn test_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "synapse-alias-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[cfg(target_os = "macos")]
    fn nm_symbols(binary: &Path) -> String {
        let output = std::process::Command::new("/usr/bin/nm")
            .args(["-a"])
            .arg(binary)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "nm -a {} failed: {}",
            binary.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).into_owned()
    }
}
