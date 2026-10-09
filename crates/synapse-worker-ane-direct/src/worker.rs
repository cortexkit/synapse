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
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    // stdin is the inherited lane lock. Never replace or close it. A separate
    // socket watcher detects disconnects even while a graph is compiling/running.
    std::thread::spawn(move || {
        let queue = unsafe { libc::kqueue() };
        if queue < 0 {
            std::process::exit(0);
        }
        let queue = unsafe { OwnedFd::from_raw_fd(queue) };
        let mut event = libc::kevent {
            ident: stream.as_raw_fd() as libc::uintptr_t,
            filter: libc::EVFILT_READ,
            flags: libc::EV_ADD | libc::EV_ENABLE,
            // Request bytes belong to the request loop. A high low-water mark
            // suppresses data-ready wakeups; EV_EOF still arrives immediately,
            // including when the owner dies with an unread request queued.
            fflags: libc::NOTE_LOWAT,
            data: libc::c_int::MAX as libc::intptr_t,
            udata: std::ptr::null_mut(),
        };
        if unsafe {
            libc::kevent(
                queue.as_raw_fd(),
                &event,
                1,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        } < 0
        {
            std::process::exit(0);
        }
        loop {
            let ready = unsafe {
                libc::kevent(
                    queue.as_raw_fd(),
                    std::ptr::null(),
                    0,
                    &mut event,
                    1,
                    std::ptr::null(),
                )
            };
            if ready > 0 && event.flags & (libc::EV_EOF | libc::EV_ERROR) != 0 {
                std::process::exit(0);
            }
            if ready < 0 && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                std::process::exit(0);
            }
        }
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
        let profile = crate::profile::LaneProfile::new();
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
                    // The module names a converted profile package
                    // "safetensors-package" for every owned engine.
                    anyhow::ensure!(
                        format == "safetensors-package",
                        "artifact_invalid: expected format safetensors-package, got {format}"
                    );
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
                            buckets: Some(crate::backend::LADDER.to_vec()),
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
                    // Acknowledge only after executable release and autorelease-pool drain.
                    model.evict(shape);
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
                        buckets: Some(crate::backend::LADDER.to_vec()),
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
        let (response, output) =
            result.unwrap_or_else(|e| (error(req_id.clone(), &e.to_string()), None));
        write_json_frame(stream, &response, max)?;
        if let Some(output) = output {
            write_frame(stream, &encode_f32_frame(&output), max)?;
        }
        profile.finish(serde_json::json!({"kind":"request", "req_id":req_id}));
    }
}
fn sequences(model: &Model, sizes: &[usize], raw: &[u8]) -> Result<Vec<f32>> {
    let mut profile = crate::profile::LaneProfile::new();
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
    profile.phase("decode_tokens");
    let mut gaps_ms = Vec::new();
    let mut envelopes_ms = Vec::new();
    let mut previous = Instant::now();
    for &size in sizes {
        let row_started = Instant::now();
        if profile.enabled() {
            gaps_ms.push(row_started.duration_since(previous).as_secs_f64() * 1000.0);
        }
        let vector = model.run(&tokens[offset..offset + size])?;
        previous = Instant::now();
        if profile.enabled() {
            envelopes_ms.push(previous.duration_since(row_started).as_secs_f64() * 1000.0);
        }
        output.extend(vector);
        offset += size;
    }
    profile.finish(serde_json::json!({"kind":"sequences", "rows":sizes.len(), "gaps_ms":gaps_ms, "row_envelopes_ms":envelopes_ms}));
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
        assert!(
            matches!(response, WorkerResponse::Pong { buckets: Some(ref buckets), .. } if buckets == &vec![128, 256, 512, 1024, 2048, 4096, 8192])
        );
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
                format: "safetensors-package".into(),
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

#[cfg(test)]
mod supervisor_tests {
    use super::watch_supervisor;
    use std::io::{Read, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};

    #[test]
    #[ignore = "subprocess helper for supervisor lifetime tests"]
    fn supervisor_watch_child() {
        let path = std::env::var_os("SYNAPSE_SUPERVISOR_TEST_SOCKET").unwrap();
        let mut stream = UnixStream::connect(path).unwrap();
        watch_supervisor(stream.try_clone().unwrap());
        stream.write_all(b"ready").unwrap();
        // Park this thread without ever reading the socket, as the real request
        // loop is stuck inside a long ANE compile. The only way this child can
        // exit is the watcher seeing the owner disconnect.
        loop {
            std::thread::park();
        }
    }

    struct ChildGuard(Child);
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }

    #[link(name = "proc")]
    unsafe extern "C" {
        fn proc_pid_rusage(
            pid: libc::c_int,
            flavor: libc::c_int,
            info: *mut libc::c_void,
        ) -> libc::c_int;
        fn mach_timebase_info(info: *mut Timebase) -> libc::c_int;
    }

    fn process_cpu_seconds(pid: u32) -> f64 {
        // RUSAGE_INFO_V0 starts with a 16-byte UUID, then user and system CPU
        // time in Mach ticks. Convert ticks to seconds with the Mach timebase;
        // on Apple Silicon a tick is not one nanosecond.
        let mut info = [0_u64; 12];
        let mut timebase = Timebase { numer: 0, denom: 0 };
        unsafe {
            assert_eq!(
                proc_pid_rusage(pid as libc::c_int, 0, info.as_mut_ptr().cast()),
                0
            );
            assert_eq!(mach_timebase_info(&mut timebase), 0);
        }
        (info[2] + info[3]) as f64 * timebase.numer as f64 / timebase.denom as f64 / 1e9
    }

    fn assert_idle_cpu(child: &mut Child, phase: &str) {
        let before = process_cpu_seconds(child.id());
        std::thread::sleep(Duration::from_secs(2));
        assert!(
            child.try_wait().unwrap().is_none(),
            "watcher exited while owner was alive"
        );
        let cpu = process_cpu_seconds(child.id()) - before;
        eprintln!("supervisor {phase}: cumulative CPU {cpu:.6}s over 2s wall");
        // A blocked watcher needs no periodic CPU budget. This bound is below
        // the measured cost of a 10 ms timed poll, with slack for registration.
        assert!(
            cpu < 0.001,
            "supervisor {phase} spun: {cpu:.6}s CPU over 2s wall"
        );
    }

    #[test]
    fn supervisor_wait_blocks_with_pending_bytes_and_detects_owner_death() {
        // A short socket path also works when Cargo runs from a long worktree.
        let path = std::env::temp_dir().join(format!("ane-supervisor-{}.sock", std::process::id()));
        let listener = UnixListener::bind(&path).unwrap();
        listener.set_nonblocking(true).unwrap();
        let mut child = ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "worker::supervisor_tests::supervisor_watch_child",
                    "--ignored",
                ])
                .env("SYNAPSE_SUPERVISOR_TEST_SOCKET", &path)
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap(),
        );
        let started = Instant::now();
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        child.0.try_wait().unwrap().is_none(),
                        "watcher helper exited before connecting"
                    );
                    assert!(
                        started.elapsed() < Duration::from_secs(10),
                        "watcher helper did not connect"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept watcher helper: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let mut ready = [0; 5];
        stream.read_exact(&mut ready).unwrap();
        assert_eq!(&ready, b"ready");
        assert_idle_cpu(&mut child.0, "idle");
        stream.write_all(b"queued request while compiling").unwrap();
        assert_idle_cpu(&mut child.0, "pending request");
        let disconnected = Instant::now();
        drop(stream);
        let status = loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                break status;
            }
            assert!(
                disconnected.elapsed() < Duration::from_secs(1),
                "owner death did not interrupt busy worker promptly"
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        eprintln!("supervisor owner-death exit: {:?}", disconnected.elapsed());
        assert!(status.success(), "watcher helper failed: {status}");
        std::fs::remove_file(path).unwrap();
    }
}
