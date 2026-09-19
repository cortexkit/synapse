#![allow(dead_code)]

use std::{
    path::{Path, PathBuf},
    process,
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde_json::Value;
use subc_protocol::Frame;
use subc_protocol::{BindIdentity, Flags, FrameType, Priority, RouteTarget};
use subc_transport::{authenticate_client, connection_file};
use subc_transport::{read_frame, write_frame};
use tokio::{
    net::TcpStream,
    process::Command,
    time::{sleep, timeout, Instant},
};

pub const MODULE_ID: &str = "synapse";
pub const SETUP_TIMEOUT: Duration = Duration::from_secs(60);
pub const READ_TIMEOUT: Duration = Duration::from_secs(30);

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static TEST_ROOT_SWEEP: OnceLock<()> = OnceLock::new();

const TEST_ROOT_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
const TEST_ROOT_PARENT: &str = "synapse-tests";

/// Install the test subscriber before a daemon starts so subc-core's wire
/// diagnostics are retained by the test harness instead of being discarded.
pub fn install_test_tracing() {
    sweep_stale_test_roots_once();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("subc_daemon=debug,info"));
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter(filter)
        .try_init();
}

fn sweep_stale_test_roots_once() {
    TEST_ROOT_SWEEP.get_or_init(|| {
        let swept = sweep_stale_test_roots_at(SystemTime::now());
        eprintln!("[test-daemon] swept {swept} stale test roots");
    });
}

fn sweep_stale_test_roots_at(now: SystemTime) -> usize {
    let parent = std::env::temp_dir().join(TEST_ROOT_PARENT);
    let Ok(entries) = std::fs::read_dir(parent) else {
        return 0;
    };
    let mut swept = 0;
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let Ok(modified) = entry.metadata().and_then(|metadata| metadata.modified()) else {
            continue;
        };
        if now
            .duration_since(modified)
            .is_ok_and(|age| age > TEST_ROOT_MAX_AGE)
            && std::fs::remove_dir_all(entry.path()).is_ok()
        {
            swept += 1;
        }
    }
    swept
}

pub fn unique_temp_dir(label: &str) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_nanos();
    // Windows can keep a child process's SQLite handles open after its test
    // harness has started termination, so cleanup may leave the old directory.
    // Include a wall-clock nonce to prevent a later process-ID reuse from
    // reopening that stale test store.
    let parent = std::env::temp_dir().join(TEST_ROOT_PARENT);
    std::fs::create_dir_all(&parent).expect("create shared test root parent");
    parent.join(format!("{label}-{}-{n}-{timestamp}", process::id()))
}

/// Give a spawned module an isolated config, data home, and singleton lease
/// scope.
///
/// Tests that need production-like settings pass them in `config_json`; all
/// other children receive an explicit empty config instead of consulting the
/// operator's HOME. The lease override keeps test modules separate from the
/// machine-wide fleet lease while preserving the production singleton policy.
///
/// The data home matters for a reason the config and lease overrides do not
/// cover: a spawned module resolves its LOG file from the data home, not from
/// its config, so without this every test run appended to the operator's live
/// log segment for today (`logs/synapse.<YYYY-MM-DD>.log`). Test traffic then
/// reads as production activity in the one surface an operator consults to see
/// what the daemon is doing. The store is unaffected either way because the
/// test daemon supplies it in HELLO_ACK.
pub fn configure_test_module_command(command: &mut Command, config_json: Option<&str>) {
    let test_root = unique_temp_dir("synapse-module-test");
    let config_path = test_root.join("synapse.jsonc");
    let lease_root = test_root.join("leases");
    let data_home = test_root.join("data");
    std::fs::create_dir_all(&lease_root).unwrap();
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::write(&config_path, config_json.unwrap_or("{}")).unwrap();
    command
        .env("SYNAPSE_CONFIG_PATH", config_path)
        .env("CORTEXKIT_LEASE_ROOT", lease_root)
        .env("XDG_DATA_HOME", data_home);
}

pub async fn connect_consumer(connection_file_path: &Path) -> TcpStream {
    let conn = connection_file::read(connection_file_path).unwrap();
    let endpoint = conn.endpoints.first().unwrap();
    let mut stream = TcpStream::connect((endpoint.host.as_str(), endpoint.port))
        .await
        .unwrap();
    authenticate_client(&mut stream, &conn, Duration::from_secs(2))
        .await
        .unwrap();
    stream
}

/// A bound route as the raw-frame tests address it: the daemon-assigned
/// channel plus the per-slot binding epoch that wire v2 requires in every
/// frame header on that route (channel 0 is fixed at epoch 0).
#[derive(Clone, Copy, Debug)]
pub struct TestRoute {
    pub channel: u16,
    pub epoch: u32,
}

pub async fn control_rpc(stream: &mut TcpStream, corr: u64, body: Value) -> Frame {
    let frame = Frame::build(
        FrameType::Request,
        Flags::new(false, Priority::Passive, false),
        0,
        0,
        corr,
        serde_json::to_vec(&body).unwrap(),
    )
    .unwrap();
    write_frame(stream, &frame).await.unwrap();
    read_until_channel0(stream, corr).await
}

async fn read_until_channel0(stream: &mut TcpStream, corr: u64) -> Frame {
    loop {
        let frame = read_frame_timeout(stream).await;
        if frame.header.channel == 0
            && matches!(frame.header.ty, FrameType::Response | FrameType::Error)
            && frame.header.corr == corr
        {
            return frame;
        }
    }
}

pub async fn read_frame_timeout(stream: &mut TcpStream) -> Frame {
    timeout(READ_TIMEOUT, async {
        read_frame(stream)
            .await
            .unwrap()
            .expect("connection should stay open")
    })
    .await
    .expect("timed out waiting for a frame")
}

pub async fn route_open(stream: &mut TcpStream, project_root: &Path, corr: u64) -> TestRoute {
    let mut last_error = String::new();
    for attempt in 0..10 {
        let target = RouteTarget::ManagementSurface {
            module_id: MODULE_ID.to_string(),
        };
        let bind_id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let identity = BindIdentity::new(
            project_root.to_path_buf(),
            format!("synapse-e2e-{}-{bind_id}", process::id()),
            format!("session-{}-{bind_id}", process::id()),
        );
        let frame = control_rpc(
            stream,
            corr + attempt,
            serde_json::json!({
                "op": "route.open",
                "target": target,
                "identity": identity,
            }),
        )
        .await;
        match frame.header.ty {
            FrameType::Response => {
                let value: Value = serde_json::from_slice(&frame.body).unwrap();
                return TestRoute {
                    channel: value["route_channel"].as_u64().unwrap() as u16,
                    epoch: value["route_epoch"].as_u64().unwrap() as u32,
                };
            }
            FrameType::Error if is_module_timeout(&frame.body) && attempt < 9 => {
                last_error = String::from_utf8_lossy(&frame.body).to_string();
                sleep(Duration::from_millis(500)).await;
            }
            _ => {
                panic!(
                    "route.open should succeed: {}",
                    String::from_utf8_lossy(&frame.body)
                );
            }
        }
    }
    panic!("route.open timed out after retries: {last_error}");
}

fn is_module_timeout(body: &[u8]) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| value["code"].as_str().map(str::to_string))
        .as_deref()
        == Some("module_timeout")
}

pub async fn route_request(
    stream: &mut TcpStream,
    route: TestRoute,
    corr: u64,
    body: Value,
) -> Value {
    let frame = raw_route_frame(stream, route, corr, body).await;
    match frame.header.ty {
        FrameType::Response => serde_json::from_slice(&frame.body).unwrap(),
        FrameType::Error => panic!(
            "route request returned error: {}",
            String::from_utf8_lossy(&frame.body)
        ),
        ty => panic!("unexpected route frame {ty:?}"),
    }
}

pub async fn raw_route_frame(
    stream: &mut TcpStream,
    route: TestRoute,
    corr: u64,
    body: Value,
) -> Frame {
    let frame = Frame::build(
        FrameType::Request,
        Flags::new(false, Priority::Interactive, false),
        route.channel,
        route.epoch,
        corr,
        serde_json::to_vec(&body).unwrap(),
    )
    .unwrap();
    write_frame(stream, &frame).await.unwrap();
    loop {
        let frame = read_frame_timeout(stream).await;
        if frame.header.corr == corr
            && matches!(frame.header.ty, FrameType::Response | FrameType::Error)
        {
            return frame;
        }
    }
}

pub async fn wait_for_catalog(stream: &mut TcpStream, module_id: &str, wait: Duration) {
    let deadline = Instant::now() + wait;
    let mut corr = 1000;
    loop {
        let frame = control_rpc(stream, corr, serde_json::json!({ "op": "catalog.list" })).await;
        assert_eq!(frame.header.ty, FrameType::Response);
        let value: Value = serde_json::from_slice(&frame.body).unwrap();
        let modules = value["modules"].as_array().cloned().unwrap_or_default();
        if modules
            .iter()
            .any(|module| module["module_id"] == module_id)
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "module {module_id} did not appear in catalog within {wait:?}"
        );
        corr += 1;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
