use std::time::Duration;
use synapse_core::{
    accept_worker_handshake_with_engine_and_protocol_version, encode_i32_frame, prepare_listener,
    read_json, write_json, write_raw, WorkerCandidate, WorkerRequest, WorkerResponse,
    DEFAULT_MAX_FRAME_BYTES,
};
use synapse_worker_vulkan::{KERNEL_REVISION, MANIFEST_DIGEST};

#[tokio::test]
async fn hosted_hello_ping_unsupported_rerank_then_ping() {
    let nonce = format!(
        "vulkan-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    // macOS socket addresses have a short limit, independent of filesystem path limits.
    #[cfg(unix)]
    let root = std::path::PathBuf::from("/tmp").join(&nonce);
    #[cfg(windows)]
    let root = std::env::temp_dir().join(&nonce);
    let (endpoint, listener) = prepare_listener(&root, &nonce).unwrap();
    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-vulkan"));
    #[cfg(unix)]
    command.arg("--socket");
    #[cfg(windows)]
    command.arg("--pipe");
    command
        .arg(endpoint)
        .arg("--nonce")
        .arg(&nonce)
        .kill_on_drop(true);
    let mut child = command.spawn().unwrap();
    let binding = synapse_core::ExpectedHelloBinding {
        manifest_digest: MANIFEST_DIGEST.into(),
        kernel_revision: KERNEL_REVISION.into(),
    };
    let mut stream = accept_worker_handshake_with_engine_and_protocol_version(
        listener,
        &nonce,
        DEFAULT_MAX_FRAME_BYTES,
        Duration::from_secs(30),
        Some("owned-vulkan"),
        None,
        Some(&binding),
    )
    .await
    .unwrap();
    for (index, id) in ["before", "after"].iter().enumerate() {
        if index == 1 {
            write_json(
                &mut stream,
                &WorkerRequest::Rerank {
                    req_id: "unsupported".into(),
                    model_ref: "not-loaded".into(),
                    query_n_tokens: 1,
                    candidates: vec![WorkerCandidate { n_tokens: 2 }],
                },
                DEFAULT_MAX_FRAME_BYTES,
            )
            .await
            .unwrap();
            write_raw(
                &mut stream,
                &encode_i32_frame(&[1, 2, 3]),
                DEFAULT_MAX_FRAME_BYTES,
            )
            .await
            .unwrap();
            let response: WorkerResponse = read_json(&mut stream, DEFAULT_MAX_FRAME_BYTES)
                .await
                .unwrap();
            assert!(
                matches!(response,WorkerResponse::Err {req_id,code,..} if req_id.as_deref()==Some("unsupported") && code=="unsupported_request")
            );
        }
        write_json(
            &mut stream,
            &WorkerRequest::Ping {
                req_id: (*id).into(),
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .await
        .unwrap();
        let response: WorkerResponse = read_json(&mut stream, DEFAULT_MAX_FRAME_BYTES)
            .await
            .unwrap();
        assert!(matches!(response,WorkerResponse::Pong {req_id,..} if req_id==*id));
    }
    write_json(
        &mut stream,
        &WorkerRequest::Shutdown {},
        DEFAULT_MAX_FRAME_BYTES,
    )
    .await
    .unwrap();
    assert!(tokio::time::timeout(Duration::from_secs(30), child.wait())
        .await
        .unwrap()
        .unwrap()
        .success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn feature_disabled_probe_is_one_json_and_exit_two() {
    if cfg!(feature = "vulkan") {
        return;
    }
    for model in [
        None,
        Some("gte-modernbert-base"),
        Some("qwen3-reranker-0.6b"),
    ] {
        let mut command =
            std::process::Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-vulkan"));
        command.arg("--probe-floor");
        if let Some(slug) = model {
            command.args(["--model", slug]);
        }
        let output = command.output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        let object: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(object["status"], "refused");
        assert_eq!(object["code"], "vulkan_no_device");
        assert!(
            object["required"]["min_device_local_bytes"]
                .as_u64()
                .unwrap()
                > 0
        );
    }
}

#[test]
fn version_reports_embedded_bindings_and_feature() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-vulkan"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("vulkan"));
    assert!(text.contains(MANIFEST_DIGEST));
    assert!(text.contains(KERNEL_REVISION));
}
