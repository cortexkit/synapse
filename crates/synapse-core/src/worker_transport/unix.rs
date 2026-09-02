use std::io;
use std::time::Duration;

use crate::worker_protocol::{WorkerHello, WorkerHelloAck, WORKER_PROTOCOL_VERSION};
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{UnixListener, UnixStream};
use tokio::time::timeout;

use crate::worker_framing::{read_json_frame, write_json_frame};

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("worker I/O: {0}")]
    Io(#[from] io::Error),
    #[error("worker protocol: {0}")]
    Protocol(String),
    #[error("worker protocol version {advertised:?} is unsupported; required {required}")]
    UnsupportedProtocolVersion {
        advertised: Option<u8>,
        required: u8,
    },
}

pub type WorkerTransportStream = UnixStream;

pub fn prepare_listener(
    runtime_dir: &std::path::Path,
    worker_id: &str,
) -> Result<(std::path::PathBuf, UnixListener), TransportError> {
    let socket_path = crate::worker_transport::worker_socket_path(runtime_dir, worker_id);
    let listener = bind_listener(&socket_path)?;
    Ok((socket_path, listener))
}

pub fn bind_listener(socket_path: &std::path::Path) -> Result<UnixListener, TransportError> {
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if socket_path.exists() {
        std::fs::remove_file(socket_path)?;
    }
    UnixListener::bind(socket_path).map_err(TransportError::from)
}

pub async fn accept_worker_handshake(
    listener: UnixListener,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
) -> Result<UnixStream, TransportError> {
    // By-value for signature parity with the Windows named-pipe variant,
    // where the listener instance IS the connection after accept. Each spawn
    // prepares a fresh listener, so single-accept ownership is the contract.
    let accept = timeout(handshake_timeout, listener.accept())
        .await
        .map_err(|_| TransportError::Protocol("worker handshake timed out".to_string()))?;
    let (mut stream, _) = accept?;
    handshake_on_stream_with_engine(
        &mut stream,
        expected_nonce,
        max_frame,
        handshake_timeout,
        None,
    )
    .await?;
    Ok(stream)
}

pub async fn accept_worker_handshake_with_engine(
    listener: UnixListener,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
    expected_engine: Option<&str>,
) -> Result<UnixStream, TransportError> {
    accept_worker_handshake_with_engine_and_protocol_version(
        listener,
        expected_nonce,
        max_frame,
        handshake_timeout,
        expected_engine,
        None,
    )
    .await
}

pub async fn accept_worker_handshake_with_engine_and_protocol_version(
    listener: UnixListener,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
    expected_engine: Option<&str>,
    required_protocol_version: Option<u8>,
) -> Result<UnixStream, TransportError> {
    let accept = timeout(handshake_timeout, listener.accept())
        .await
        .map_err(|_| TransportError::Protocol("worker handshake timed out".to_string()))?;
    let (mut stream, _) = accept?;
    handshake_on_stream_with_engine_and_protocol_version(
        &mut stream,
        expected_nonce,
        max_frame,
        handshake_timeout,
        expected_engine,
        required_protocol_version,
    )
    .await?;
    Ok(stream)
}

pub async fn handshake_on_stream<S>(
    stream: &mut S,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
) -> Result<(), TransportError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    handshake_on_stream_with_engine(stream, expected_nonce, max_frame, handshake_timeout, None)
        .await
}

pub async fn handshake_on_stream_with_engine<S>(
    stream: &mut S,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
    expected_engine: Option<&str>,
) -> Result<(), TransportError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    handshake_on_stream_with_engine_and_protocol_version(
        stream,
        expected_nonce,
        max_frame,
        handshake_timeout,
        expected_engine,
        None,
    )
    .await
}

pub async fn handshake_on_stream_with_engine_and_protocol_version<S>(
    stream: &mut S,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
    expected_engine: Option<&str>,
    required_protocol_version: Option<u8>,
) -> Result<(), TransportError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let hello: serde_json::Value = timeout(handshake_timeout, read_json_frame(stream, max_frame))
        .await
        .map_err(|_| TransportError::Protocol("worker HELLO timed out".to_string()))??;
    let advertised_protocol_version = hello
        .get("protocol_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|version| u8::try_from(version).ok());
    let hello: WorkerHello = serde_json::from_value(hello)
        .map_err(|error| TransportError::Protocol(format!("invalid worker HELLO: {error}")))?;
    if hello.v != WORKER_PROTOCOL_VERSION || hello.nonce != expected_nonce {
        return Err(TransportError::Protocol(format!(
            "rejected worker HELLO v={} nonce_match={}",
            hello.v,
            hello.nonce == expected_nonce
        )));
    }
    if expected_engine.is_some_and(|engine| hello.engine.engine != engine) {
        return Err(TransportError::Protocol(format!(
            "rejected worker HELLO engine={}, expected {}",
            hello.engine.engine,
            expected_engine.unwrap_or_default()
        )));
    }
    if required_protocol_version
        .is_some_and(|required| advertised_protocol_version != Some(required))
    {
        return Err(TransportError::UnsupportedProtocolVersion {
            advertised: advertised_protocol_version,
            required: required_protocol_version.unwrap_or_default(),
        });
    }
    let accepted_frame = max_frame.min(hello.max_frame);
    let mut ack = serde_json::to_value(WorkerHelloAck {
        v: WORKER_PROTOCOL_VERSION,
        accept: true,
        max_frame: accepted_frame,
    })
    .map_err(|error| TransportError::Protocol(format!("encode worker HELLO_ACK: {error}")))?;
    if let Some(protocol_version) = required_protocol_version {
        ack["protocol_version"] = serde_json::Value::from(protocol_version);
    }
    write_json_frame(stream, &ack, accepted_frame).await?;
    Ok(())
}

pub async fn read_json<T: DeserializeOwned, S: AsyncRead + Unpin>(
    stream: &mut S,
    max_frame: u32,
) -> Result<T, TransportError> {
    read_json_frame(stream, max_frame)
        .await
        .map_err(TransportError::from)
}

pub async fn write_json<T: Serialize, S: AsyncWrite + Unpin>(
    stream: &mut S,
    value: &T,
    max_frame: u32,
) -> Result<(), TransportError> {
    write_json_frame(stream, value, max_frame)
        .await
        .map_err(TransportError::from)
}

pub async fn read_raw<S: AsyncRead + Unpin>(
    stream: &mut S,
    max_frame: u32,
) -> Result<Vec<u8>, TransportError> {
    crate::worker_framing::read_frame(stream, max_frame)
        .await
        .map_err(TransportError::from)
}

pub async fn write_raw<S: AsyncWrite + Unpin>(
    stream: &mut S,
    bytes: &[u8],
    max_frame: u32,
) -> Result<(), TransportError> {
    crate::worker_framing::write_frame(stream, bytes, max_frame)
        .await
        .map_err(TransportError::from)
}
