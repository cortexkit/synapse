//! Hosted protocol checks against immutable, already-built candidate binaries.

use std::{path::PathBuf, time::Duration};
use synapse_core::{
    accept_worker_handshake_with_engine_and_protocol_version, encode_i32_frame, prepare_listener,
    read_json, write_json, write_raw, WorkerCandidate, WorkerRequest, WorkerResponse,
    WorkerSequence, DEFAULT_MAX_FRAME_BYTES as MAX,
};

#[tokio::test]
#[ignore = "requires release-candidate binaries in CANDIDATE_BIN_DIR"]
async fn candidate_hosted_protocol_matrix() {
    let Some(root) = std::env::var_os("CANDIDATE_BIN_DIR") else {
        eprintln!("skipping candidate protocol matrix: CANDIDATE_BIN_DIR is unset");
        return;
    };
    let root = PathBuf::from(root);
    assert!(root.is_absolute(), "CANDIDATE_BIN_DIR must be absolute");
    let suffix = if cfg!(windows) { ".exe" } else { "" };
    let scratch = std::env::temp_dir().join(format!("candidate-module-{}", std::process::id()));
    assert!(std::process::Command::new(
        synapse_core::dev_binary::ckdev_binary(root.join(format!("ck-synapse{suffix}")), &scratch)
            .unwrap()
    )
    .arg("--version")
    .status()
    .unwrap()
    .success());
    let mut workers = vec![("llama", "llama.cpp-worker", true)];
    if cfg!(target_os = "macos") {
        workers.extend([
            ("decode", "owned-metal-decode", false),
            ("ane", "ane-coreml-worker", false),
            ("ane-direct", "ane-direct-worker", true),
        ]);
    } else {
        workers.extend([
            ("cuda", "owned-cuda", true),
            ("vulkan", "owned-vulkan", true),
        ]);
    }
    for (worker, engine, unsupported) in workers {
        println!("hosted protocol: {worker}");
        let nonce = format!("candidate-{}-{worker}", std::process::id());
        let runtime = std::env::temp_dir().join(&nonce);
        let (endpoint, listener) = prepare_listener(&runtime, &nonce).unwrap();
        let mut command = tokio::process::Command::new(
            synapse_core::dev_binary::ckdev_binary(
                root.join(format!("ck-synapse-worker-{worker}{suffix}")),
                &runtime,
            )
            .unwrap(),
        );
        if worker == "ane" {
            command.env(
                "SYNAPSE_ANE_SWIFT_WORKER",
                synapse_core::dev_binary::ckdev_binary(
                    root.join("ck-synapse-worker-ane-swift"),
                    &runtime,
                )
                .unwrap(),
            );
        }
        command
            .arg(if cfg!(windows) { "--pipe" } else { "--socket" })
            .arg(endpoint)
            .arg("--nonce")
            .arg(&nonce)
            .kill_on_drop(true);
        let mut child = command.spawn().unwrap();
        let mut stream = accept_worker_handshake_with_engine_and_protocol_version(
            listener,
            &nonce,
            MAX,
            Duration::from_secs(30),
            Some(engine),
            None,
            None,
        )
        .await
        .unwrap();
        for index in 0..=usize::from(unsupported) {
            if index == 1 {
                let request = if worker == "llama" {
                    WorkerRequest::RerankSequences {
                        req_id: "unsupported".into(),
                        model_ref: "not-loaded".into(),
                        sequences: vec![WorkerSequence { n_tokens: 2 }],
                    }
                } else {
                    WorkerRequest::Rerank {
                        req_id: "unsupported".into(),
                        model_ref: "not-loaded".into(),
                        query_n_tokens: 1,
                        candidates: vec![WorkerCandidate { n_tokens: 1 }],
                    }
                };
                write_json(&mut stream, &request, MAX).await.unwrap();
                write_raw(&mut stream, &encode_i32_frame(&[1, 2]), MAX)
                    .await
                    .unwrap();
                let response: WorkerResponse =
                    tokio::time::timeout(Duration::from_secs(30), read_json(&mut stream, MAX))
                        .await
                        .unwrap()
                        .unwrap();
                assert!(
                    matches!(response, WorkerResponse::Err { code, .. } if code == "unsupported_request")
                );
            }
            write_json(
                &mut stream,
                &WorkerRequest::Ping {
                    req_id: format!("ping-{index}"),
                },
                MAX,
            )
            .await
            .unwrap();
            let response: WorkerResponse =
                tokio::time::timeout(Duration::from_secs(30), read_json(&mut stream, MAX))
                    .await
                    .unwrap()
                    .unwrap();
            assert!(
                matches!(response, WorkerResponse::Pong { req_id, .. } if req_id == format!("ping-{index}"))
            );
        }
        write_json(&mut stream, &WorkerRequest::Shutdown {}, MAX)
            .await
            .unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(30), child.wait())
            .await
            .unwrap()
            .unwrap()
            .success());
        let _ = std::fs::remove_dir_all(runtime);
    }
    let _ = std::fs::remove_dir_all(scratch);
}
