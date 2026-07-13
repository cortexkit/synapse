#![allow(dead_code)]

use std::{
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use serde_json::Value;
use subc_core::{read_frame, write_frame, Frame};
use subc_protocol::{BindIdentity, Flags, FrameType, Priority, RouteTarget};
use subc_transport::{authenticate_client, connection_file};
use tokio::{
    net::TcpStream,
    time::{sleep, timeout, Instant},
};

pub const MODULE_ID: &str = "synapse";
pub const SETUP_TIMEOUT: Duration = Duration::from_secs(10);
pub const READ_TIMEOUT: Duration = Duration::from_secs(30);

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn unique_temp_dir(label: &str) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{label}-{}-{n}", process::id()))
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
        let identity = BindIdentity {
            project_root: project_root.to_path_buf(),
            harness: format!("synapse-e2e-{}-{bind_id}", process::id()),
            session: format!("session-{}-{bind_id}", process::id()),
        };
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
