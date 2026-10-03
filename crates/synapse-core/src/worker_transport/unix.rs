use std::io;
use std::time::Duration;

use crate::worker_protocol::{
    check_hello_binding, ExpectedHelloBinding, HelloBindingMismatch, WorkerHello, WorkerHelloAck,
    WORKER_PROTOCOL_VERSION,
};
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
    /// An owned worker's HELLO named a different (or no) manifest digest or
    /// kernel revision than the host's lane requires.
    #[error("rejected worker HELLO: {0}")]
    HelloBindingMismatch(#[from] HelloBindingMismatch),
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
        None,
    )
    .await
}

/// Accepts one worker connection and validates its HELLO: protocol version,
/// nonce, engine identity, the owned-decode envelope version when required,
/// and, when `expected_binding` is given, an owned worker's manifest digest
/// and kernel revision (see [`check_hello_binding`]).
pub async fn accept_worker_handshake_with_engine_and_protocol_version(
    listener: UnixListener,
    expected_nonce: &str,
    max_frame: u32,
    handshake_timeout: Duration,
    expected_engine: Option<&str>,
    required_protocol_version: Option<u8>,
    expected_binding: Option<&ExpectedHelloBinding>,
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
        expected_binding,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker_protocol::{
        ERR_KERNEL_REVISION_MISMATCH, ERR_MANIFEST_MISMATCH, OWNED_WORKER_HELLO_ENGINES,
    };
    use crate::EngineIdentity;

    const NONCE: &str = "0123456789abcdef";

    fn binding() -> ExpectedHelloBinding {
        ExpectedHelloBinding {
            manifest_digest: "d".repeat(64),
            kernel_revision: "kernel-rev-a".to_string(),
        }
    }

    fn hello_json(
        engine: &str,
        manifest: Option<&str>,
        revision: Option<&str>,
    ) -> serde_json::Value {
        serde_json::to_value(WorkerHello {
            v: WORKER_PROTOCOL_VERSION,
            nonce: NONCE.to_string(),
            engine: EngineIdentity {
                engine: engine.to_string(),
                version: "test".to_string(),
                build_flags: Default::default(),
            },
            pid: 1,
            max_frame: 4096,
            manifest_digest: manifest.map(str::to_owned),
            kernel_revision: revision.map(str::to_owned),
        })
        .unwrap()
    }

    /// Runs the host side of the handshake against a HELLO already written by
    /// the worker end of an in-memory pipe, and returns the host's verdict.
    async fn handshake(
        hello: serde_json::Value,
        expected_engine: Option<&str>,
        required_protocol_version: Option<u8>,
        expected_binding: Option<&ExpectedHelloBinding>,
    ) -> Result<(), TransportError> {
        let (mut host, mut worker) = tokio::io::duplex(64 * 1024);
        write_json_frame(&mut worker, &hello, 4096).await.unwrap();
        let verdict = handshake_on_stream_with_engine_and_protocol_version(
            &mut host,
            NONCE,
            4096,
            Duration::from_secs(1),
            expected_engine,
            required_protocol_version,
            expected_binding,
        )
        .await;
        if verdict.is_ok() {
            let ack: WorkerHelloAck = read_json_frame(&mut worker, 4096).await.unwrap();
            assert!(ack.accept);
            assert_eq!(ack.v, WORKER_PROTOCOL_VERSION);
        }
        verdict
    }

    fn binding_code(error: TransportError) -> &'static str {
        match error {
            TransportError::HelloBindingMismatch(mismatch) => mismatch.code,
            other => panic!("expected a HELLO binding refusal, got {other}"),
        }
    }

    #[tokio::test]
    async fn bound_host_refuses_an_owned_worker_with_a_missing_or_different_binding() {
        let expected = binding();
        let digest = expected.manifest_digest.clone();
        for engine in OWNED_WORKER_HELLO_ENGINES {
            handshake(
                hello_json(engine, Some(&digest), Some("kernel-rev-a")),
                Some(engine),
                None,
                Some(&expected),
            )
            .await
            .expect("matching binding is accepted");
            for (manifest, revision, code) in [
                (None, Some("kernel-rev-a"), ERR_MANIFEST_MISMATCH),
                (
                    Some("e".repeat(64)),
                    Some("kernel-rev-a"),
                    ERR_MANIFEST_MISMATCH,
                ),
                (Some(digest.clone()), None, ERR_KERNEL_REVISION_MISMATCH),
                (
                    Some(digest.clone()),
                    Some("kernel-rev-b"),
                    ERR_KERNEL_REVISION_MISMATCH,
                ),
            ] {
                let error = handshake(
                    hello_json(engine, manifest.as_deref(), revision),
                    Some(engine),
                    None,
                    Some(&expected),
                )
                .await
                .expect_err("mismatched binding is refused");
                assert_eq!(binding_code(error), code, "{engine}");
            }
        }
    }

    #[tokio::test]
    async fn unbound_host_accepts_an_owned_worker_without_binding_fields() {
        handshake(
            hello_json("owned-cuda", None, None),
            Some("owned-cuda"),
            None,
            None,
        )
        .await
        .expect("lanes without an expected binding keep today's handshake");
        let error = handshake(
            hello_json("owned-cuda", None, None),
            Some("owned-cuda"),
            None,
            Some(&binding()),
        )
        .await
        .expect_err("the same worker is refused once the lane expects a binding");
        assert_eq!(binding_code(error), ERR_MANIFEST_MISMATCH);
    }

    #[tokio::test]
    async fn non_owned_workers_pass_a_bound_host_with_both_fields_absent() {
        for engine in ["llama.cpp-worker", "ane-coreml-worker"] {
            handshake(
                hello_json(engine, None, None),
                Some(engine),
                None,
                Some(&binding()),
            )
            .await
            .expect("non-owned workers are not manifest-bound");
        }
        let mut decode = hello_json("owned-metal-decode", None, None);
        decode["protocol_version"] = serde_json::Value::from(2);
        handshake(
            decode,
            Some("owned-metal-decode"),
            Some(2),
            Some(&binding()),
        )
        .await
        .expect("decode HELLO at v2 passes with neither binding field");
    }

    #[tokio::test]
    async fn wrong_engine_identity_and_old_protocol_are_rejected() {
        let error = handshake(
            hello_json("llama.cpp-worker", None, None),
            Some("owned-cuda"),
            None,
            None,
        )
        .await
        .expect_err("engine identity must match");
        assert!(
            matches!(error, TransportError::Protocol(message) if message.contains("engine=llama.cpp-worker"))
        );

        let mut old = hello_json("llama.cpp-worker", None, None);
        old["v"] = serde_json::Value::from(1);
        let error = handshake(old, Some("llama.cpp-worker"), None, None)
            .await
            .expect_err("v1 HELLO is refused");
        assert!(matches!(error, TransportError::Protocol(message) if message.contains("v=1")));
    }
}
