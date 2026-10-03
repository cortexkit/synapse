use std::io;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};
use tokio::time::timeout;

use crate::worker_framing::{read_frame, read_json_frame, write_frame, write_json_frame};
use crate::worker_protocol::{
    check_hello_binding, ExpectedHelloBinding, HelloBindingMismatch, WorkerHello, WorkerHelloAck,
    WORKER_PROTOCOL_VERSION,
};

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
    /// An owned worker's HELLO named a different (or no) manifest digest or
    /// kernel revision than the host's lane requires.
    #[error("rejected worker HELLO: {0}")]
    HelloBindingMismatch(#[from] HelloBindingMismatch),
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
    server: NamedPipeServer,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
    expected_engine: Option<&str>,
) -> Result<NamedPipeServer, TransportError> {
    accept_worker_handshake_with_engine_and_protocol_version(
        server,
        expected_nonce,
        max_frame,
        handshake_timeout,
        expected_engine,
        None,
        None,
    )
    .await
}

/// Accepts one worker connection and validates its HELLO: protocol version,
/// nonce, engine identity, the owned-decode envelope version when required,
/// and, when `expected_binding` is given, an owned worker's manifest digest
/// and kernel revision (see [`check_hello_binding`]).
pub async fn accept_worker_handshake_with_engine_and_protocol_version(
    mut server: NamedPipeServer,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
    expected_engine: Option<&str>,
    required_protocol_version: Option<u8>,
    expected_binding: Option<&ExpectedHelloBinding>,
) -> Result<NamedPipeServer, TransportError> {
    timeout(handshake_timeout, server.connect())
        .await
        .map_err(|_| TransportError::Protocol("worker handshake timed out".to_string()))??;
    handshake_on_stream_with_engine_and_protocol_version(
        &mut server,
        expected_nonce,
        max_frame,
        handshake_timeout,
        expected_engine,
        required_protocol_version,
        expected_binding,
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
    handshake_on_stream_with_engine_and_protocol_version(
        stream,
        expected_nonce,
        max_frame,
        handshake_timeout,
        expected_engine,
        None,
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
    expected_binding: Option<&ExpectedHelloBinding>,
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
    check_hello_binding(&hello, expected_binding)?;
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
            manifest_digest: None,
            kernel_revision: None,
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

    /// Runs the host side of the handshake against a HELLO already written by
    /// the worker end of an in-memory pipe, and returns the host's verdict.
    async fn bound_handshake(
        engine: &str,
        manifest: Option<&str>,
        revision: Option<&str>,
        expected_binding: Option<&ExpectedHelloBinding>,
    ) -> Result<(), TransportError> {
        let nonce = "0123456789abcdef";
        let (mut host, mut worker) = tokio::io::duplex(64 * 1024);
        let hello = WorkerHello {
            v: WORKER_PROTOCOL_VERSION,
            nonce: nonce.to_string(),
            engine: EngineIdentity {
                engine: engine.to_string(),
                version: "test".to_string(),
                build_flags: Default::default(),
            },
            pid: 1,
            max_frame: 4096,
            manifest_digest: manifest.map(str::to_owned),
            kernel_revision: revision.map(str::to_owned),
        };
        write_json_frame(&mut worker, &hello, 4096).await.unwrap();
        handshake_on_stream_with_engine_and_protocol_version(
            &mut host,
            nonce,
            4096,
            Duration::from_secs(1),
            Some(engine),
            None,
            expected_binding,
        )
        .await
    }

    #[tokio::test]
    async fn pipe_handshake_enforces_the_owned_worker_binding() {
        use crate::worker_protocol::{ERR_KERNEL_REVISION_MISMATCH, ERR_MANIFEST_MISMATCH};

        let expected = ExpectedHelloBinding {
            manifest_digest: "d".repeat(64),
            kernel_revision: "kernel-rev-a".to_string(),
        };
        let digest = expected.manifest_digest.clone();
        bound_handshake("owned-vulkan", None, None, None)
            .await
            .expect("lanes without an expected binding keep today's handshake");
        bound_handshake(
            "owned-vulkan",
            Some(&digest),
            Some("kernel-rev-a"),
            Some(&expected),
        )
        .await
        .expect("matching binding is accepted");
        bound_handshake("llama.cpp-worker", None, None, Some(&expected))
            .await
            .expect("non-owned workers are not manifest-bound");
        for (manifest, revision, code) in [
            (None, Some("kernel-rev-a"), ERR_MANIFEST_MISMATCH),
            (Some(digest.as_str()), None, ERR_KERNEL_REVISION_MISMATCH),
            (
                Some(digest.as_str()),
                Some("kernel-rev-b"),
                ERR_KERNEL_REVISION_MISMATCH,
            ),
        ] {
            match bound_handshake("owned-cuda", manifest, revision, Some(&expected)).await {
                Err(TransportError::HelloBindingMismatch(mismatch)) => {
                    assert_eq!(mismatch.code, code)
                }
                other => panic!("expected a {code} refusal, got {other:?}"),
            }
        }
    }
}
