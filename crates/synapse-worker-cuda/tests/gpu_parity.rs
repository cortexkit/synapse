#![cfg(all(unix, feature = "cuda"))]
use std::{
    collections::BTreeMap,
    os::unix::net::UnixListener,
    path::PathBuf,
    process::{Child, Command},
    time::{Duration, Instant},
};
use synapse_core::{
    worker_framing_sync::{read_frame, read_json_frame, write_frame, write_json_frame},
    WorkerHello, WorkerHelloAck, WorkerPooling, WorkerRequest, WorkerResponse, WorkerSequence,
    WorkerTokenItem, DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
};
use synapse_engine_cuda::manifest::{manifest_digest, Profile};
struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU, CUDA 13.2 runtimes and SYNAPSE_CUDA_TEST_PACKAGES inside this worktree"]
fn converted_cuda_packages_match_padded_goldens_and_ignore_request_pooling() {
    let root = PathBuf::from(
        std::env::var_os("SYNAPSE_CUDA_TEST_PACKAGES").expect("converted package directory"),
    );
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    assert!(root.canonicalize().unwrap().starts_with(&repo));
    let socket = std::env::temp_dir().join(format!("cu-gpu-{}.sock", std::process::id()));
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut worker = Worker(
        Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-cuda"))
            .args([
                "--socket",
                socket.to_str().unwrap(),
                "--nonce",
                "gpu-parity",
            ])
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    let mut stream = loop {
        match listener.accept() {
            Ok((s, _)) => break s,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(
                    worker.0.try_wait().unwrap().is_none(),
                    "worker exited before HELLO"
                );
                assert!(start.elapsed() < Duration::from_secs(15));
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("{e}"),
        }
    };
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(300)))
        .unwrap();
    let hello: WorkerHello = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
    assert_eq!(hello.manifest_digest, Some(manifest_digest()));
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
    for slug in [
        "gte-modernbert-base",
        "qwen3-embedding-0.6b",
        "gte-reranker-modernbert-base",
        "qwen3-reranker-0.6b",
    ] {
        let p = Profile::select(&format!("{slug}.owned-cuda"), None).unwrap();
        let package = root.join(slug);
        assert!(!package.join("config.json").exists());
        write_json_frame(
            &mut stream,
            &WorkerRequest::Load {
                req_id: "load".into(),
                artifact_path: package.to_str().unwrap().into(),
                artifact_digest: p.package_digest.clone(),
                format: "safetensors-package".into(),
                runtime_config: BTreeMap::from([
                    ("profile".into(), p.id.clone()),
                    ("operation".into(), p.operation().into()),
                ]),
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        let response: WorkerResponse =
            read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
        let WorkerResponse::Loaded {
            model_ref, dims, ..
        } = response
        else {
            panic!("{slug} load: {response:?}")
        };
        let fixture = repo.join(format!(
            "bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json"
        ));
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
        let cases = &fixture["cases"].as_array().unwrap()[..3];
        let lengths: Vec<_> = cases
            .iter()
            .map(|c| c["input_ids"].as_array().unwrap().len())
            .collect();
        assert!(
            lengths.iter().any(|n| *n != lengths[0]),
            "fixture must exercise padding"
        );
        let mut raw = Vec::new();
        for case in cases {
            for id in case["input_ids"].as_array().unwrap() {
                raw.extend_from_slice(&(id.as_i64().unwrap() as i32).to_le_bytes());
            }
        }
        let mut baseline = None;
        for pooling in [WorkerPooling::Mean, WorkerPooling::Last] {
            let request = if p.operation() == "embed" {
                WorkerRequest::EmbedBatch {
                    req_id: "infer".into(),
                    model_ref: model_ref.clone(),
                    pooling,
                    normalize: false,
                    items: lengths
                        .iter()
                        .enumerate()
                        .map(|(i, n)| WorkerTokenItem {
                            id: i.to_string(),
                            n_tokens: *n,
                        })
                        .collect(),
                }
            } else {
                WorkerRequest::RerankSequences {
                    req_id: "infer".into(),
                    model_ref: model_ref.clone(),
                    sequences: lengths
                        .iter()
                        .map(|n| WorkerSequence { n_tokens: *n })
                        .collect(),
                }
            };
            write_json_frame(&mut stream, &request, DEFAULT_MAX_FRAME_BYTES).unwrap();
            write_frame(&mut stream, &raw, DEFAULT_MAX_FRAME_BYTES).unwrap();
            let response: WorkerResponse =
                read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
            assert!(
                matches!(
                    response,
                    WorkerResponse::Vectors { .. } | WorkerResponse::Scores { .. }
                ),
                "{response:?}"
            );
            let raw = read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
            let values: Vec<f32> = raw
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(values.len(), cases.len() * dims);
            if let Some(baseline) = &baseline {
                assert_eq!(
                    &values, baseline,
                    "EMBED_BATCH pooling must not override the manifest"
                );
            } else {
                baseline = Some(values.clone());
            }
            for (row, case) in cases.iter().enumerate() {
                if p.operation() == "rerank" {
                    assert!(
                        (values[row] - case["output"].as_f64().unwrap() as f32).abs() <= 0.02,
                        "{slug} padded score row {row}"
                    );
                } else {
                    let actual = &values[row * dims..(row + 1) * dims];
                    let expected = case["output"].as_array().unwrap();
                    assert_eq!(expected.len(), dims);
                    let norm = actual.iter().map(|v| v * v).sum::<f32>().sqrt();
                    assert!((norm - 1.0).abs() < 1e-3);
                    let dot = actual
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| a * b.as_f64().unwrap() as f32)
                        .sum::<f32>();
                    assert!(dot >= 0.999, "{slug} padded cosine row {row}: {dot}");
                }
            }
        }
        write_json_frame(
            &mut stream,
            &WorkerRequest::Unload {
                req_id: "unload".into(),
                model_ref,
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        let _: WorkerResponse = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
    }
    write_json_frame(
        &mut stream,
        &WorkerRequest::Shutdown {},
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    assert!(worker.0.wait().unwrap().success());
    std::fs::remove_file(socket).unwrap();
}
