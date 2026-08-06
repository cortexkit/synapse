use std::io;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};
use tokio::time::timeout;

use crate::worker_framing::{read_frame, read_json_frame, write_frame, write_json_frame};
use crate::worker_protocol::{WorkerHello, WorkerHelloAck, WORKER_PROTOCOL_VERSION};

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("worker I/O: {0}")]
    Io(#[from] io::Error),
    #[error("worker protocol: {0}")]
    Protocol(String),
}

pub type WorkerTransportStream = NamedPipeServer;

pub fn prepare_listener(
    _runtime_dir: &std::path::Path,
    worker_id: &str,
) -> Result<(String, NamedPipeServer), TransportError> {
    let pipe_name = crate::worker_transport::worker_pipe_name(worker_id);
    let server = bind_listener(&pipe_name)?;
    Ok((pipe_name, server))
}

pub fn bind_listener(pipe_name: &str) -> Result<NamedPipeServer, TransportError> {
    ServerOptions::new()
        .first_pipe_instance(true)
        .create(pipe_name)
        .map_err(TransportError::from)
}

pub async fn accept_worker_handshake(
    mut server: NamedPipeServer,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
) -> Result<NamedPipeServer, TransportError> {
    // A named-pipe server instance becomes the connection once a client
    // connects (unlike a unix listener, which yields a separate stream), so
    // this takes the listener by value and returns it as the stream.
    timeout(handshake_timeout, server.connect())
        .await
        .map_err(|_| TransportError::Protocol("worker handshake timed out".to_string()))??;
    handshake_on_stream_with_engine(
        &mut server,
        expected_nonce,
        max_frame,
        handshake_timeout,
        None,
    )
    .await?;
    Ok(server)
}

pub async fn accept_worker_handshake_with_engine(
    mut server: NamedPipeServer,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
    expected_engine: Option<&str>,
) -> Result<NamedPipeServer, TransportError> {
    timeout(handshake_timeout, server.connect())
        .await
        .map_err(|_| TransportError::Protocol("worker handshake timed out".to_string()))??;
    handshake_on_stream_with_engine(
        &mut server,
        expected_nonce,
        max_frame,
        handshake_timeout,
        expected_engine,
    )
    .await?;
    Ok(server)
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
    read_frame(stream, max_frame)
        .await
        .map_err(TransportError::from)
}

pub async fn write_raw<S: AsyncWrite + Unpin>(
    stream: &mut S,
    bytes: &[u8],
    max_frame: u32,
) -> Result<(), TransportError> {
    write_frame(stream, bytes, max_frame)
        .await
        .map_err(TransportError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker_transport::worker_pipe_name;
    use crate::EngineIdentity;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::SystemTime;
    use tokio::net::windows::named_pipe::ClientOptions;

    fn test_nonce() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let value = now ^ u64::from(std::process::id()) ^ COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{value:016x}")
    }

    #[tokio::test]
    async fn pipe_framing_round_trip() {
        let worker_id = format!("framing-test-{}", test_nonce());
        let pipe_name = worker_pipe_name(&worker_id);
        let server = bind_listener(&pipe_name).expect("create pipe server");
        let expected_nonce = test_nonce();
        let client_nonce = expected_nonce.clone();
        let max_frame = 4096_u32;

        let server_task = tokio::spawn(async move {
            server.connect().await.expect("client connect");
            let mut stream = server;
            handshake_on_stream(
                &mut stream,
                &expected_nonce,
                max_frame,
                Duration::from_secs(5),
            )
            .await
            .expect("handshake");
            write_json_frame(
                &mut stream,
                &serde_json::json!({"type":"PONG","req_id":"1"}),
                max_frame,
            )
            .await
            .expect("write pong");
            read_frame(&mut stream, max_frame).await.expect("read raw")
        });

        let mut client = ClientOptions::new()
            .open(&pipe_name)
            .expect("open client pipe");
        let hello = WorkerHello {
            v: WORKER_PROTOCOL_VERSION,
            nonce: client_nonce,
            engine: EngineIdentity {
                engine: "test".to_string(),
                version: "0".to_string(),
                build_flags: Default::default(),
            },
            pid: 1,
            max_frame,
        };
        write_json_frame(&mut client, &hello, max_frame)
            .await
            .expect("write hello");
        let ack: WorkerHelloAck = read_json_frame(&mut client, max_frame)
            .await
            .expect("read ack");
        assert!(ack.accept);
        let _: serde_json::Value = read_json_frame(&mut client, max_frame)
            .await
            .expect("read pong");
        write_frame(&mut client, b"tensor-bytes", max_frame)
            .await
            .expect("write raw");
        let raw = server_task.await.expect("server task");
        assert_eq!(raw, b"tensor-bytes");
    }
}
