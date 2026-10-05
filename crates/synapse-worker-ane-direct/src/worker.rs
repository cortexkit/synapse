use crate::backend::{Model, Profile, REVISION};
use anyhow::{Context, Result};
use clap::Parser;
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use synapse_core::worker_framing_sync::{
    read_frame, read_json_frame, write_frame, write_json_frame,
};
use synapse_core::{
    decode_i32_frame, encode_f32_frame, EngineIdentity, WorkerHello, WorkerHelloAck, WorkerRequest,
    WorkerResponse, DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    socket: PathBuf,
    #[arg(long)]
    nonce: String,
}

fn private_api() -> Result<()> {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    let available = AVAILABLE.get_or_init(|| unsafe {
        // Keep the framework open for the process lifetime: the Objective-C
        // classes and compiled models can outlive an individual LOAD request.
        let handle = libc::dlopen(
            c"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
                .as_ptr(),
            libc::RTLD_NOW | libc::RTLD_GLOBAL,
        );
        if handle.is_null() {
            return false;
        }
        unsafe extern "C" {
            fn objc_getClass(name: *const libc::c_char) -> *mut libc::c_void;
        }
        [
            c"_ANEInMemoryModel",
            c"_ANEInMemoryModelDescriptor",
            c"_ANERequest",
            c"_ANEIOSurfaceObject",
        ]
        .iter()
        .all(|name| !objc_getClass(name.as_ptr()).is_null())
    });
    anyhow::ensure!(*available, "ane_private_api_unavailable");
    Ok(())
}
fn floor(probe: impl FnOnce() -> Result<()>) -> serde_json::Value {
    match probe() {
        Ok(()) => {
            serde_json::json!({"status":"ok","required":{"private_api":"_ANEInMemoryModel"},"observed":{"private_api":true}})
        }
        Err(_) => {
            serde_json::json!({"status":"refused","code":"ane_private_api_unavailable","required":{"private_api":"_ANEInMemoryModel"},"observed":null})
        }
    }
}
fn hello(nonce: String) -> WorkerHello {
    WorkerHello {
        v: WORKER_PROTOCOL_VERSION,
        nonce,
        pid: std::process::id(),
        max_frame: DEFAULT_MAX_FRAME_BYTES,
        engine: EngineIdentity {
            engine: "ane-direct-worker".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            build_flags: BTreeMap::from([
                ("precision".into(), "f16".into()),
                ("kernel_revision".into(), REVISION.into()),
            ]),
        },
        manifest_digest: Some(env!("ANE_MANIFEST_DIGEST").into()),
        kernel_revision: Some(REVISION.into()),
    }
}
pub fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|s| s == "--version") {
        println!(
            "ck-synapse-worker-ane-direct {} manifest_digest={} kernel_revision={}",
            env!("CARGO_PKG_VERSION"),
            env!("ANE_MANIFEST_DIGEST"),
            REVISION
        );
        return Ok(());
    }
    if args.iter().any(|s| s == "--probe-floor") {
        let envelope = floor(private_api);
        println!("{envelope}");
        if envelope["status"] != "ok" {
            std::process::exit(2);
        }
        return Ok(());
    }
    let args = Args::parse();
    let mut stream = std::os::unix::net::UnixStream::connect(&args.socket)?;
    watch_supervisor(stream.try_clone()?);
    write_json_frame(&mut stream, &hello(args.nonce), DEFAULT_MAX_FRAME_BYTES)?;
    let ack: WorkerHelloAck = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES)?;
    anyhow::ensure!(
        ack.accept
            && ack.v == WORKER_PROTOCOL_VERSION
            && ack.max_frame <= DEFAULT_MAX_FRAME_BYTES
            && ack.max_frame > 0,
        "handshake_refused"
    );
    request_loop(&mut stream, ack.max_frame, private_api)
}
fn watch_supervisor(stream: std::os::unix::net::UnixStream) {
    use std::os::fd::AsRawFd;
    // stdin is the inherited lane lock. Never replace or close it. A separate
    // socket watcher detects disconnects even while a graph is compiling/running.
    std::thread::spawn(move || loop {
        let mut byte = 0u8;
        let n = unsafe {
            libc::recv(
                stream.as_raw_fd(),
                (&mut byte as *mut u8).cast(),
                1,
                libc::MSG_PEEK | libc::MSG_DONTWAIT,
            )
        };
        if n == 0 {
            std::process::exit(0);
        }
        if n < 0 {
            let error = io::Error::last_os_error();
            if !matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) {
                std::process::exit(0);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    });
}
fn error(req_id: Option<String>, message: &str) -> WorkerResponse {
    let code = message.split(':').next().unwrap_or("inference_failed");
    WorkerResponse::Err {
        req_id,
        code: code.into(),
        msg: message.into(),
    }
}
fn os_build() -> Result<String> {
    let output = synapse_core::without_launch_nonce(std::process::Command::new("sw_vers"))
        .arg("-buildVersion")
        .output()?;
    anyhow::ensure!(output.status.success(), "os_build_unavailable");
    Ok(String::from_utf8(output.stdout)?.trim().into())
}
fn request_loop<S: Read + Write>(
    stream: &mut S,
    max: u32,
    mut probe: impl FnMut() -> Result<()>,
) -> Result<()> {
    let mut models: BTreeMap<String, Model> = BTreeMap::new();
    loop {
        let frame = match read_frame(stream, max) {
            Ok(frame) => frame,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let request: WorkerRequest = serde_json::from_slice(&frame)?;
        let raw = if request.carries_raw_frame() {
            Some(read_frame(stream, max)?)
        } else {
            None
        };
        if matches!(request, WorkerRequest::Shutdown {}) {
            return Ok(());
        }
        let req_id = request.req_id().map(str::to_owned);
        let result: Result<(WorkerResponse, Option<Vec<f32>>)> = (|| {
            Ok(match request {
                WorkerRequest::Load {
                    req_id,
                    artifact_path,
                    artifact_digest,
                    format,
                    runtime_config,
                } => {
                    probe()?;
                    anyhow::ensure!(format == "safetensors", "artifact_invalid");
                    let id = runtime_config.get("profile").context("model_unsupported")?;
                    let operation = runtime_config
                        .get("operation")
                        .context("operation_mismatch")?;
                    let profile = Profile::select(id, operation)?;
                    let model_ref = format!("ane-direct:{id}:{artifact_digest}");
                    let start = Instant::now();
                    let model = Model::load(profile, Path::new(&artifact_path), &artifact_digest)?;
                    let dims = model.profile.model["output"]["dimension"]
                        .as_u64()
                        .context("model_unsupported")? as usize;
                    models.insert(model_ref.clone(), model);
                    (
                        WorkerResponse::Loaded {
                            req_id,
                            model_ref,
                            dims,
                            cold_load_ms: start.elapsed().as_millis() as u64,
                            buckets: None,
                        },
                        None,
                    )
                }
                WorkerRequest::AneAdmitShape {
                    req_id,
                    model_ref,
                    shape,
                } => {
                    let already = models
                        .get(&model_ref)
                        .is_some_and(|m| m.resident.contains_key(&shape));
                    anyhow::ensure!(
                        already || models.values().map(|m| m.resident.len()).sum::<usize>() < 8,
                        "ane_residency_limit"
                    );
                    let inventory = models
                        .get_mut(&model_ref)
                        .context("model_not_loaded")?
                        .admit(shape, &os_build()?)?;
                    (WorkerResponse::Admitted { req_id, inventory }, None)
                }
                WorkerRequest::AneEvictShape {
                    req_id,
                    model_ref,
                    shape,
                } => {
                    let model = models.get_mut(&model_ref).context("model_not_loaded")?;
                    // Drop unloads every executable before acknowledging eviction.
                    drop(model.resident.remove(&shape));
                    (WorkerResponse::Evicted { req_id }, None)
                }
                WorkerRequest::EmbedBatch {
                    req_id,
                    model_ref,
                    items,
                    ..
                } => {
                    let model = models.get(&model_ref).context("model_not_loaded")?;
                    anyhow::ensure!(model.profile.operation() == "embed", "operation_mismatch");
                    let vectors = sequences(
                        model,
                        &items.iter().map(|i| i.n_tokens).collect::<Vec<_>>(),
                        raw.as_deref().unwrap(),
                    )?;
                    (
                        WorkerResponse::Vectors {
                            req_id,
                            dims: model.profile.n("hidden_size"),
                            n: items.len(),
                        },
                        Some(vectors),
                    )
                }
                WorkerRequest::RerankSequences {
                    req_id,
                    model_ref,
                    sequences: items,
                } => {
                    let model = models.get(&model_ref).context("model_not_loaded")?;
                    anyhow::ensure!(model.profile.operation() == "rerank", "operation_mismatch");
                    let scores = sequences(
                        model,
                        &items.iter().map(|i| i.n_tokens).collect::<Vec<_>>(),
                        raw.as_deref().unwrap(),
                    )?;
                    (WorkerResponse::Scores { req_id }, Some(scores))
                }
                WorkerRequest::Ping { req_id } => (
                    WorkerResponse::Pong {
                        req_id,
                        rss_mb: 0,
                        models_loaded: models.len(),
                        placement_share: None,
                        buckets: None,
                        ane_resident_shapes: Some(
                            models
                                .iter()
                                .map(|(key, model)| {
                                    (key.clone(), model.resident.keys().copied().collect())
                                })
                                .collect(),
                        ),
                    },
                    None,
                ),
                WorkerRequest::Unload { req_id, model_ref } => {
                    models.remove(&model_ref).context("model_not_loaded")?;
                    (WorkerResponse::Unloaded { req_id }, None)
                }
                WorkerRequest::Shutdown {} => {
                    return Ok((
                        WorkerResponse::Unloaded {
                            req_id: String::new(),
                        },
                        None,
                    ))
                }
                ref other => (WorkerResponse::unsupported_request(other), None),
            })
        })();
        let (response, output) = result.unwrap_or_else(|e| (error(req_id, &e.to_string()), None));
        write_json_frame(stream, &response, max)?;
        if let Some(output) = output {
            write_frame(stream, &encode_f32_frame(&output), max)?;
        }
    }
}
fn sequences(model: &Model, sizes: &[usize], raw: &[u8]) -> Result<Vec<f32>> {
    let tokens = decode_i32_frame(raw)?;
    let total = sizes
        .iter()
        .try_fold(0usize, |n, size| n.checked_add(*size))
        .context("invalid_request")?;
    anyhow::ensure!(
        total == tokens.len() && !sizes.is_empty(),
        "invalid_request"
    );
    let tokens: Vec<u32> = tokens
        .into_iter()
        .map(u32::try_from)
        .collect::<Result<_, _>>()?;
    let mut output = Vec::new();
    let mut offset = 0;
    for &size in sizes {
        output.extend(model.run(&tokens[offset..offset + size])?);
        offset += size;
    }
    Ok(output)
}

#[cfg(test)]
pub(crate) fn test_private_api() -> Result<()> {
    private_api()
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_core::worker_framing_sync::write_json_frame;
    struct Duplex {
        input: io::Cursor<Vec<u8>>,
        output: Vec<u8>,
    }
    impl Read for Duplex {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            self.input.read(b)
        }
    }
    impl Write for Duplex {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.output.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn hello_and_ping_do_not_resolve_private_framework() {
        let hello = hello("test".into());
        assert_eq!(
            hello.manifest_digest.as_deref(),
            Some(env!("ANE_MANIFEST_DIGEST"))
        );
        assert_eq!(hello.kernel_revision.as_deref(), Some(REVISION));
        let mut input = Vec::new();
        write_json_frame(
            &mut input,
            &WorkerRequest::Ping {
                req_id: "ping".into(),
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        let mut stream = Duplex {
            input: io::Cursor::new(input),
            output: Vec::new(),
        };
        request_loop(&mut stream, DEFAULT_MAX_FRAME_BYTES, || {
            anyhow::bail!("ane_private_api_unavailable")
        })
        .unwrap();
        let response: WorkerResponse =
            read_json_frame(&mut io::Cursor::new(stream.output), DEFAULT_MAX_FRAME_BYTES).unwrap();
        assert!(matches!(response, WorkerResponse::Pong { .. }));
    }
    #[test]
    fn unsupported_rerank_drains_raw_frame_then_pongs() {
        let mut input = Vec::new();
        write_json_frame(
            &mut input,
            &WorkerRequest::Rerank {
                req_id: "rerank".into(),
                model_ref: "missing".into(),
                query_n_tokens: 1,
                candidates: vec![],
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        write_frame(&mut input, &1i32.to_le_bytes(), DEFAULT_MAX_FRAME_BYTES).unwrap();
        write_json_frame(
            &mut input,
            &WorkerRequest::Ping {
                req_id: "ping".into(),
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        let mut stream = Duplex {
            input: io::Cursor::new(input),
            output: Vec::new(),
        };
        request_loop(&mut stream, DEFAULT_MAX_FRAME_BYTES, || {
            panic!("unexpected symbol lookup")
        })
        .unwrap();
        let mut output = io::Cursor::new(stream.output);
        let response: WorkerResponse =
            read_json_frame(&mut output, DEFAULT_MAX_FRAME_BYTES).unwrap();
        assert!(
            matches!(response, WorkerResponse::Err { code, .. } if code == "unsupported_request")
        );
        let response: WorkerResponse =
            read_json_frame(&mut output, DEFAULT_MAX_FRAME_BYTES).unwrap();
        assert!(matches!(response, WorkerResponse::Pong { .. }));
    }
    #[test]
    fn missing_private_api_is_typed_at_floor_and_load() {
        assert_eq!(
            floor(|| anyhow::bail!("missing"))["code"],
            "ane_private_api_unavailable"
        );
        let mut input = Vec::new();
        write_json_frame(
            &mut input,
            &WorkerRequest::Load {
                req_id: "load".into(),
                artifact_path: "absent".into(),
                artifact_digest: "missing".into(),
                format: "safetensors".into(),
                runtime_config: BTreeMap::new(),
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        let mut stream = Duplex {
            input: io::Cursor::new(input),
            output: Vec::new(),
        };
        request_loop(&mut stream, DEFAULT_MAX_FRAME_BYTES, || {
            anyhow::bail!("ane_private_api_unavailable")
        })
        .unwrap();
        let response: WorkerResponse =
            read_json_frame(&mut io::Cursor::new(stream.output), DEFAULT_MAX_FRAME_BYTES).unwrap();
        assert!(
            matches!(response, WorkerResponse::Err { code, .. } if code == "ane_private_api_unavailable")
        );
    }
}
