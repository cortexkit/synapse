#![cfg(unix)]
use std::{
    collections::BTreeMap,
    os::unix::net::UnixListener,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use synapse_core::{
    worker_framing_sync::{read_json_frame, write_frame, write_json_frame},
    WorkerHello, WorkerHelloAck, WorkerRequest, WorkerResponse, DEFAULT_MAX_FRAME_BYTES,
    WORKER_PROTOCOL_VERSION,
};

#[test]
fn hosted_worker_hello_ping_unsupported_rerank_ping() {
    let path = std::env::temp_dir().join(format!(
        "cu-{}-{}.sock",
        std::process::id(),
        std::thread::current().name().unwrap().len()
    ));
    let listener = UnixListener::bind(&path).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-cuda"))
        .args(["--socket", path.to_str().unwrap(), "--nonce", "hosted-test"])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let start = Instant::now();
    let mut stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(
                    child.try_wait().unwrap().is_none(),
                    "worker exited before HELLO (possible load-time CUDA dependency)"
                );
                if start.elapsed() > Duration::from_secs(15) {
                    child.kill().unwrap();
                    panic!("worker did not reach HELLO");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("{e}"),
        }
    };
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let hello: WorkerHello = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
    assert_eq!(hello.nonce, "hosted-test");
    assert_eq!(
        hello.manifest_digest.as_deref(),
        Some(synapse_engine_cuda::manifest::manifest_digest().as_str())
    );
    assert!(hello.kernel_revision.is_some());
    write_json_frame(
        &mut stream,
        &WorkerHelloAck {
            v: WORKER_PROTOCOL_VERSION,
            accept: true,
            max_frame: DEFAULT_MAX_FRAME_BYTES,
        },
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    let ping = |stream: &mut std::os::unix::net::UnixStream, id: &str| {
        write_json_frame(
            stream,
            &WorkerRequest::Ping { req_id: id.into() },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        assert!(
            matches!(read_json_frame::<_,WorkerResponse>(stream,DEFAULT_MAX_FRAME_BYTES).unwrap(),WorkerResponse::Pong {req_id,..} if req_id==id)
        );
    };
    ping(&mut stream, "before");
    write_json_frame(
        &mut stream,
        &WorkerRequest::Rerank {
            req_id: "unsupported".into(),
            model_ref: "none".into(),
            query_n_tokens: 1,
            candidates: vec![],
        },
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    write_frame(&mut stream, &42i32.to_le_bytes(), DEFAULT_MAX_FRAME_BYTES).unwrap();
    assert!(
        matches!(read_json_frame::<_,WorkerResponse>(&mut stream,DEFAULT_MAX_FRAME_BYTES).unwrap(),WorkerResponse::Err {code,..} if code=="unsupported_request")
    );
    ping(&mut stream, "after");
    write_json_frame(
        &mut stream,
        &WorkerRequest::Load {
            req_id: "load".into(),
            artifact_path: "missing-model".into(),
            artifact_digest: "sha256:abc".into(),
            format: "safetensors-package".into(),
            runtime_config: BTreeMap::new(),
        },
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    let _: WorkerResponse = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
    write_json_frame(
        &mut stream,
        &WorkerRequest::Shutdown {},
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    assert!(child.wait().unwrap().success());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn probe_is_one_typed_envelope_without_artifact_access() {
    let output = Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-cuda"))
        .args(["--probe-floor", "--model", "gte-modernbert-base"])
        .output()
        .unwrap();
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.lines().count(), 1);
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value.as_object().unwrap().len(), 4);
    assert_eq!(value["required"]["driver_api"], 13020);
    assert_eq!(
        output.status.code(),
        Some(if value["status"] == "ok" { 0 } else { 2 })
    );
}
