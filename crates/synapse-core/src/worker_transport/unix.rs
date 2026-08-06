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
    let accept = timeout(handshake_timeout, listener.accept())
        .await
        .map_err(|_| TransportError::Protocol("worker handshake timed out".to_string()))?;
    let (mut stream, _) = accept?;
    handshake_on_stream_with_engine(
        &mut stream,
        expected_nonce,
        max_frame,
        handshake_timeout,
        expected_engine,
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
    let hello: WorkerHello = timeout(handshake_timeout, read_json_frame(stream, max_frame))
        .await
        .map_err(|_| TransportError::Protocol("worker HELLO timed out".to_string()))??;
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
    let accepted_frame = max_frame.min(hello.max_frame);
    write_json_frame(
        stream,
        &WorkerHelloAck {
            v: WORKER_PROTOCOL_VERSION,
            accept: true,
            max_frame: accepted_frame,
        },
        accepted_frame,
    )
    .await?;
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
