//! Tests for the shared helpers in `tests/common/mod.rs`. That module is
//! compiled into more than one test binary, so its own tests live here to run
//! once, in this binary only.

use std::{
    process,
    sync::atomic::Ordering,
    time::{Duration, SystemTime},
};

use crate::common::*;

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[tokio::test]
async fn whole_process_without_home_holds_lease_at_native_data_home() {
    use cortexkit_lease::{FileLeaseStore, LeaseError, LeaseKey, LeaseStore};
    use subc_transport::{
        generate_daemon_id, generate_key, write_atomic, ConnectionInfo, Endpoint, SCHEMA_VERSION,
    };
    use tokio::{io::AsyncReadExt, net::TcpListener, process::Command, time::timeout};

    let root = unique_temp_dir("native-data-home");
    let data_home = root.join("data");
    std::fs::create_dir_all(&data_home).unwrap();
    let config_path = root.join("synapse.jsonc");
    std::fs::write(&config_path, "{}").unwrap();
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let connection_path = root.join("subc-conn.json");
    write_atomic(
        &connection_path,
        &ConnectionInfo {
            schema: SCHEMA_VERSION,
            endpoints: vec![Endpoint {
                host: "127.0.0.1".to_string(),
                port: listener.local_addr().unwrap().port(),
            }],
            key: generate_key().unwrap(),
            daemon_id: generate_daemon_id().unwrap(),
            pid: process::id(),
            daemon_ver: "test-native-data-home".to_string(),
            wire_version: Some(subc_protocol::PROTOCOL_VERSION),
        },
    )
    .unwrap();
    let binary =
        synapse_core::dev_binary::ckdev_binary(env!("CARGO_BIN_EXE_ck-synapse"), &root).unwrap();
    let mut command = Command::new(binary);
    command
        .arg("--subc")
        .arg(&connection_path)
        .env_remove("HOME")
        .env_remove("CORTEXKIT_LEASE_ROOT")
        .env("SUBC_MODULE_ID", MODULE_ID)
        .env("SYNAPSE_CONFIG_PATH", config_path)
        .env("XDG_DATA_HOME", &data_home)
        .env("LOCALAPPDATA", &data_home)
        .env("APPDATA", root.join("roaming"))
        .env("USERPROFILE", root.join("profile"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let mut child = synapse_core::without_launch_nonce_tokio(command)
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();

    // Connecting happens after lease acquisition. Hold the TCP connection
    // without authentication or HELLO_ACK so profile and store boot stay out
    // of this test, and check the actual lease while the child is alive.
    let connected = timeout(Duration::from_secs(10), async {
        tokio::select! {
            connection = listener.accept() => connection.map_err(|error| error.to_string()),
            status = child.wait() => Err(format!("module exited before connecting: {status:?}")),
        }
    })
    .await;
    let lease_root = data_home.join("cortexkit").join("leases");
    let lease = if matches!(&connected, Ok(Ok(_))) {
        match FileLeaseStore::new(&lease_root).acquire(&LeaseKey::new(
            MODULE_ID,
            "file",
            "singleton",
        )) {
            Err(LeaseError::Held { .. }) => Ok(()),
            Err(error) => Err(format!("lease check refused: {error}")),
            Ok(_handle) => Err("native data-home lease was not held".to_string()),
        }
    } else {
        Err(format!("module did not connect: {connected:?}"))
    };

    // Reap before asserting, including on a startup refusal or timeout.
    let _ = child.start_kill();
    timeout(Duration::from_secs(10), child.wait())
        .await
        .unwrap()
        .unwrap();
    let mut output = Vec::new();
    timeout(Duration::from_secs(10), stdout.read_to_end(&mut output))
        .await
        .unwrap()
        .unwrap();
    timeout(Duration::from_secs(10), stderr.read_to_end(&mut output))
        .await
        .unwrap()
        .unwrap();
    drop(connected);
    let _ = std::fs::remove_dir_all(&root);
    assert!(
        lease.is_ok(),
        "{lease:?}; module output: {}",
        String::from_utf8_lossy(&output)
    );
}

#[cfg(unix)]
#[test]
fn exit_status_description_distinguishes_exit_code_and_signal() {
    let exited = std::process::Command::new("/bin/sh")
        .args(["-c", "exit 7"])
        .status()
        .unwrap();
    assert_eq!(describe_exit_status(&exited), "exited with status 7");

    let signaled = std::process::Command::new("/bin/sh")
        .args(["-c", "kill -KILL $$"])
        .status()
        .unwrap();
    assert_eq!(describe_exit_status(&signaled), "killed by signal 9");
}

/// Open a directory handle that `set_times` can act on. On Windows a plain
/// `File::open` of a directory fails with "Access is denied" twice over:
/// a directory handle needs FILE_FLAG_BACKUP_SEMANTICS, which std does not
/// set, and writing timestamps needs FILE_WRITE_ATTRIBUTES access, which a
/// read-only open does not carry (the first fix opened the handle and then
/// failed on set_times with the same message). The sweep itself is
/// unaffected (it reads mtime through `metadata()`); only this test, which
/// ages a directory by hand, needs the handle.
fn open_directory_for_times(path: &std::path::Path) -> std::fs::File {
    let mut options = std::fs::OpenOptions::new();
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        const FILE_WRITE_ATTRIBUTES: u32 = 0x0100;
        options
            .access_mode(FILE_WRITE_ATTRIBUTES)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS);
    }
    #[cfg(not(windows))]
    options.read(true);
    options.open(path).unwrap()
}

#[test]
fn sweeps_only_stale_test_root_children() {
    let suffix = format!(
        "{}-{}",
        process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let parent = std::env::temp_dir().join(TEST_ROOT_PARENT);
    let old = parent.join(format!("old-{suffix}"));
    let young = parent.join(format!("young-{suffix}"));
    let outside = std::env::temp_dir().join(format!("outside-{suffix}"));
    for path in [&old, &young, &outside] {
        std::fs::create_dir_all(path).unwrap();
    }
    let old_mtime = SystemTime::now() - Duration::from_secs(25 * 60 * 60);
    for path in [&old, &outside] {
        open_directory_for_times(path)
            .set_times(std::fs::FileTimes::new().set_modified(old_mtime))
            .unwrap();
    }

    sweep_stale_test_roots_at(SystemTime::now());

    assert!(!old.exists(), "the 25-hour-old child must be swept");
    assert!(young.exists(), "the young child must remain");
    assert!(outside.exists(), "a sibling outside the parent must remain");
    let _ = std::fs::remove_dir_all(young);
    let _ = std::fs::remove_dir_all(outside);
}
