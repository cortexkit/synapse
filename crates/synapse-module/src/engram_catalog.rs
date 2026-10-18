//! The module-owned Engram descriptor for data under Synapse's native data root.

#[cfg(unix)]
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// These rebuild entries keep their durable names but declare no path: current
// certification evidence lives in the database and decode-worker scratch lives
// outside the module data root.
const CATALOG: &[u8] = br#"{"schema_version":1,"module_id":"synapse","entries":[{"entry_id":"synapse/store","class":"portable","mechanism":"whole-db","path":"store.db","writer_interaction":"backup-api-live"},{"entry_id":"synapse/certification","class":"rebuild","mechanism":"filetree"},{"entry_id":"synapse/decode-worker","class":"rebuild","mechanism":"filetree"},{"entry_id":"synapse/logs","class":"excluded","mechanism":"filetree","path":"logs","reason":"Operational logs are disposable diagnostics, not module data."}]}"#;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Write the fixed descriptor atomically; `false` means its bytes were unchanged.
pub(crate) fn write(root: &Path) -> io::Result<bool> {
    fs::create_dir_all(root)?;
    let destination = root.join("engram-catalog.json");
    match fs::symlink_metadata(&destination) {
        Ok(metadata) => {
            if !metadata.file_type().is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "catalog destination is not a regular file",
                ));
            }
            if fs::read(&destination)? == CATALOG {
                set_owner_only(&destination)?;
                return Ok(false);
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }

    let temporary = temporary_path(&destination);
    let result = write_and_publish(&temporary, &destination);
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map(|()| true)
}

fn temporary_path(destination: &Path) -> PathBuf {
    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let filename = destination
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    destination.with_file_name(format!(".{filename}.{}.{}.tmp", std::process::id(), id))
}

fn write_and_publish(temporary: &Path, destination: &Path) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    set_creation_mode(&mut options);
    let mut file = options.open(temporary)?;
    file.write_all(CATALOG)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temporary, destination)?;
    set_owner_only(destination)?;
    sync_parent(destination.parent().unwrap_or_else(|| Path::new(".")))
}

#[cfg(unix)]
fn set_creation_mode(options: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    options.mode(0o600);
}

#[cfg(not(unix))]
fn set_creation_mode(_options: &mut OpenOptions) {}

#[cfg(unix)]
fn set_owner_only(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
}

#[cfg(windows)]
fn set_owner_only(_path: &Path) -> io::Result<()> {
    // Files created under LOCALAPPDATA inherit that directory's user-only ACL.
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn set_owner_only(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn sync_parent(parent: &Path) -> io::Result<()> {
    File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent(_parent: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "synapse-engram-catalog-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn writes_stable_bytes_and_keeps_the_legacy_entry_ids() {
        let root = temporary_root("stable");
        let destination = root.join("engram-catalog.json");

        assert!(write(&root).expect("first write succeeds"));
        let first = fs::read(&destination).expect("read first catalog");
        assert!(!write(&root).expect("unchanged write succeeds"));
        let second = fs::read(&destination).expect("read second catalog");
        assert_eq!(first, second);
        assert_eq!(first, CATALOG);

        let parsed: serde_json::Value = serde_json::from_slice(&second).unwrap();
        let entry_ids = parsed["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["entry_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            &entry_ids[..3],
            [
                "synapse/store",
                "synapse/certification",
                "synapse/decode-worker"
            ]
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(destination).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn an_unwritable_data_root_returns_an_error() {
        use std::os::unix::fs::PermissionsExt;

        let root = temporary_root("unwritable");
        fs::create_dir_all(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).unwrap();
        let result = write(&root);
        let _ = fs::set_permissions(&root, fs::Permissions::from_mode(0o700));
        let _ = fs::remove_dir_all(root);

        assert!(result.is_err(), "a root without write permission must fail");
    }
}
