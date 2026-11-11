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
    decode_i32_frame, encode_f32_frame, EngineIdentity, WorkerHello, WorkerHelloAck,
    WorkerProgressPhase, WorkerRequest, WorkerResponse, DEFAULT_MAX_FRAME_BYTES,
    WORKER_PROTOCOL_VERSION,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    socket: PathBuf,
    #[arg(long)]
    nonce: String,
}

pub(crate) fn private_api() -> Result<()> {
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
        packed: Some(synapse_core::PackedCapability::tight256()),
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

pub(crate) fn request_error(req_id: Option<String>, failure: &anyhow::Error) -> WorkerResponse {
    let mut response = error(req_id, &failure.to_string());
    if let WorkerResponse::Err { code, msg, .. } = &mut response {
        // Layer context is a diagnostic, not an error class. Inspect the typed
        // cause even when compilation added one or more contextual messages.
        if failure.chain().any(|cause| {
            cause
                .downcast_ref::<ane::Error>()
                .is_some_and(ane::Error::is_resource_refusal)
        }) {
            *code = "ane_resources_exhausted".into();
        }
        *msg = format!("{failure:#}");
    }
    response
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
    let mut packed = BTreeMap::<String, crate::multirow::tight::TightProgram>::new();
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
        let mut progress_writer = synapse_core::worker_framing_sync::ProgressWriter::new(
            stream,
            req_id.as_deref().unwrap_or_default(),
            max,
        );
        let mut progress = |phase| {
            progress_writer
                .completed(phase)
                .map_err(anyhow::Error::from)
        };
        let profile = crate::profile::LaneProfile::new();
        let result: Result<Vec<(WorkerResponse, Option<Vec<u8>>)>> = (|| {
            let response = match request {
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
                    if operation == "transcribe" {
                        crate::parakeet::validate_load(&profile, max)?;
                    }
                    let model_ref = format!("ane-direct:{id}:{artifact_digest}");
                    let start = Instant::now();
                    let model = Model::load(profile, Path::new(&artifact_path), &artifact_digest)?;
                    let dims = if operation == "transcribe" {
                        0
                    } else {
                        model.profile.model["output"]["dimension"]
                            .as_u64()
                            .context("model_unsupported")? as usize
                    };
                    // Which rungs aned already holds compiled, so the module
                    // predicts a fast load for them instead of a compile.
                    let check_started = Instant::now();
                    let cached = model.cached_rungs();
                    // The module logs each worker stderr line at INFO.
                    eprintln!(
                        "ane-direct compile cache check: model_ref={model_ref} cached={cached:?} checked={} total_ms={:.1}",
                        crate::backend::LADDER.len(),
                        check_started.elapsed().as_secs_f64() * 1000.0
                    );
                    models.insert(model_ref.clone(), model);
                    (
                        loaded_response(
                            req_id,
                            model_ref,
                            dims,
                            start.elapsed().as_millis() as u64,
                            Some(crate::backend::LADDER.to_vec()),
                            Some(cached),
                            operation == "transcribe",
                        ),
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
                    let admission = models
                        .get_mut(&model_ref)
                        .context("model_not_loaded")?
                        .admit_progress(shape, &os_build()?, &mut || {
                            progress(WorkerProgressPhase::Setup)
                        })?;
                    (
                        WorkerResponse::Admitted {
                            req_id,
                            inventory: admission.inventory,
                            cache_hit_layers: Some(admission.cache_hit_layers),
                            compiled_layers: Some(admission.compiled_layers),
                        },
                        None,
                    )
                }
                WorkerRequest::AneAdmitPacked {
                    req_id,
                    model_ref,
                    program_revision,
                } => {
                    anyhow::ensure!(
                        program_revision == synapse_core::PACKED_PROGRAM_REVISION,
                        "unsupported_request"
                    );
                    let model = models.get(&model_ref).context("model_not_loaded")?;
                    if !packed.contains_key(&model_ref) {
                        let program = model.compile_tight_packing_progress(
                            synapse_core::PACKED_WIDTH,
                            &mut || progress(WorkerProgressPhase::Setup),
                        )?;
                        packed.insert(model_ref.clone(), program);
                    }
                    let inventory = synapse_core::AnePlacementInventory {
                        executables: (0..packed[&model_ref].executable_count())
                            .map(|layer| synapse_core::AneExecutable {
                                id: format!("tight256-r{program_revision}-layer{layer}"),
                                layers: vec![layer as u32],
                            })
                            .collect(),
                        cpu_stages: vec!["embedding_gather".into(), "final_rms_l2".into()],
                    };
                    (
                        WorkerResponse::Admitted {
                            req_id,
                            inventory,
                            cache_hit_layers: None,
                            compiled_layers: None,
                        },
                        None,
                    )
                }
                WorkerRequest::AneEvictPacked {
                    req_id,
                    model_ref,
                    program_revision,
                } => {
                    anyhow::ensure!(
                        program_revision == synapse_core::PACKED_PROGRAM_REVISION,
                        "unsupported_request"
                    );
                    drop(packed.remove(&model_ref));
                    (WorkerResponse::Evicted { req_id }, None)
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
                        &mut || progress(WorkerProgressPhase::Inference),
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
                        &mut || progress(WorkerProgressPhase::Inference),
                    )?;
                    (WorkerResponse::Scores { req_id }, Some(scores))
                }
                WorkerRequest::Transcribe {
                    req_id,
                    model_ref,
                    mel_frames,
                    valid_mel_frames,
                    observe,
                } => {
                    let raw = raw.as_deref().context("invalid_mel_frame_length")?;
                    validate_transcribe_frame(raw, mel_frames, max)?;
                    let model = models.get(&model_ref).context("model_not_loaded")?;
                    let mel = synapse_core::decode_f32_frame(raw)?;
                    let output = model.transcribe_progress(
                        &mel,
                        mel_frames,
                        valid_mel_frames,
                        observe,
                        &mut || progress(WorkerProgressPhase::Inference),
                    )?;
                    return transcribe_responses(&req_id, output, observe, max);
                }
                WorkerRequest::EmbedPacked {
                    req_id,
                    model_ref,
                    program_revision,
                    items,
                    passes,
                } => {
                    anyhow::ensure!(
                        program_revision == synapse_core::PACKED_PROGRAM_REVISION,
                        "unsupported_request"
                    );
                    let model = models.get(&model_ref).context("model_not_loaded")?;
                    let program = packed.get(&model_ref).context("shape_not_admitted")?;
                    let vectors = packed_sequences_progress(
                        model,
                        program,
                        &items.iter().map(|i| i.n_tokens).collect::<Vec<_>>(),
                        &passes,
                        raw.as_deref().unwrap(),
                        &mut || progress(WorkerProgressPhase::Inference),
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
                    drop(packed.remove(&model_ref));
                    models.remove(&model_ref).context("model_not_loaded")?;
                    (WorkerResponse::Unloaded { req_id }, None)
                }
                WorkerRequest::Shutdown {} => unreachable!("handled before dispatch"),
                ref other => (WorkerResponse::unsupported_request(other), None),
            };
            Ok(vec![(
                response.0,
                response.1.map(|values| encode_f32_frame(&values)),
            )])
        })();
        if progress_writer.failed() {
            anyhow::bail!("progress stream write failed");
        }
        let responses = result.unwrap_or_else(|e| vec![(request_error(req_id.clone(), &e), None)]);
        for (response, output) in responses {
            write_json_frame(stream, &response, max)?;
            if let Some(output) = output {
                write_frame(stream, &output, max)?;
            }
        }
        profile.finish(serde_json::json!({"kind":"request", "req_id":req_id}));
    }
}

fn validate_transcribe_frame(raw: &[u8], mel_frames: usize, max: u32) -> Result<()> {
    anyhow::ensure!(
        mel_frames.checked_mul(512) == Some(raw.len()),
        "invalid_mel_frame_length"
    );
    anyhow::ensure!(raw.len() <= max as usize, "transcribe_max_frame_exceeded");
    Ok(())
}

fn loaded_response(
    req_id: String,
    model_ref: String,
    dims: usize,
    cold_load_ms: u64,
    buckets: Option<Vec<usize>>,
    cached: Option<Vec<usize>>,
    transcribe: bool,
) -> WorkerResponse {
    let mut response =
        WorkerResponse::loaded(req_id, model_ref, dims, cold_load_ms, buckets, cached);
    if transcribe {
        if let WorkerResponse::Loaded {
            transcribe_revision,
            ..
        } = &mut response
        {
            *transcribe_revision = Some(synapse_core::TranscribeRevision {
                kernel_revision: synapse_core::ANE_DIRECT_TRANSCRIBE_KERNEL_REVISION.into(),
                decode_rule: synapse_speech::DECODE_RULE_REVISION.into(),
            });
        }
    }
    response
}

fn transcribe_responses(
    req_id: &str,
    output: crate::parakeet::Output,
    observe: bool,
    max: u32,
) -> Result<Vec<(WorkerResponse, Option<Vec<u8>>)>> {
    let mut frames = Vec::new();
    if observe {
        frames.push((
            WorkerResponse::EncoderOutput {
                req_id: req_id.into(),
                frames: output.frames,
            },
            Some(encode_f32_frame(&output.encoder)),
        ));
        let trace: Vec<_> = output
            .decoded
            .trace
            .iter()
            .map(|s| synapse_core::DecodeTraceRecord {
                frame_index: s.frame_index as i32,
                token_top1: s.token_top1 as i32,
                token_top2: s.token_top2 as i32,
                duration_top1: s.duration_top1 as i32,
                duration_top2: s.duration_top2 as i32,
                token_logit1: s.token_logit1,
                token_logit2: s.token_logit2,
                duration_logit1: s.duration_logit1,
                duration_logit2: s.duration_logit2,
                flags: u32::from(s.emitted) | (u32::from(s.cap_hit) << 1),
            })
            .collect();
        frames.push((
            WorkerResponse::DecodeTrace {
                req_id: req_id.into(),
                steps: trace.len(),
            },
            Some(synapse_core::encode_decode_trace_frame(&trace)),
        ));
    }
    let triples: Vec<_> = output
        .decoded
        .triples
        .iter()
        .map(|s| synapse_core::TranscriptRecord {
            token_id: s[0] as i32,
            frame_index: s[1] as i32,
            duration: s[2] as i32,
        })
        .collect();
    frames.push((
        WorkerResponse::Transcript {
            req_id: req_id.into(),
            valid_encoder_frames: output.frames,
            count: triples.len(),
        },
        Some(synapse_core::encode_transcript_frame(&triples)),
    ));
    // Refuse before writing any observation header, so a frame-size failure
    // never leaves the connection halfway through a multi-frame response.
    for (response, raw) in &frames {
        let raw = raw.as_deref().unwrap();
        anyhow::ensure!(raw.len() <= max as usize, "transcribe_max_frame_exceeded");
        response.validate_raw_frame(raw)?;
    }
    Ok(frames)
}

#[cfg(test)]
mod transcribe_tests {
    use super::*;
    struct Stream {
        input: io::Cursor<Vec<u8>>,
        output: Vec<u8>,
    }
    impl Read for Stream {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.input.read(bytes)
        }
    }
    impl Write for Stream {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.output.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn transcribe_frame_length_is_exact_and_bounded() {
        assert!(validate_transcribe_frame(&vec![0; 1024], 2, 1024).is_ok());
        for bytes in [1023, 1025] {
            assert!(validate_transcribe_frame(&vec![0; bytes], 2, 2048).is_err());
        }
        assert!(validate_transcribe_frame(&vec![0; 1024], 2, 1023).is_err());
        assert!(validate_transcribe_frame(&[], usize::MAX, u32::MAX).is_err());
    }

    #[test]
    fn only_transcribe_loaded_reports_both_compiled_revisions() {
        let loaded = |speech| loaded_response("r".into(), "m".into(), 0, 1, None, None, speech);
        let legacy = serde_json::to_value(loaded(false)).unwrap();
        assert!(legacy.get("transcribe_revision").is_none());
        let speech = serde_json::to_value(loaded(true)).unwrap();
        assert_eq!(
            speech["transcribe_revision"]["kernel_revision"],
            synapse_core::ANE_DIRECT_TRANSCRIBE_KERNEL_REVISION
        );
        assert_eq!(
            speech["transcribe_revision"]["decode_rule"],
            synapse_speech::DECODE_RULE_REVISION
        );
    }

    fn output() -> crate::parakeet::Output {
        crate::parakeet::Output {
            frames: 1,
            encoder: vec![0.25; 1024],
            decoded: synapse_speech::DecodeOutput {
                triples: vec![[7, 0, 2]],
                trace: vec![synapse_speech::TraceStep {
                    frame_index: 0,
                    token_top1: 7,
                    token_top2: 8192,
                    duration_top1: 2,
                    duration_top2: 1,
                    token_logit1: 4.0,
                    token_logit2: 3.0,
                    duration_logit1: 2.0,
                    duration_logit2: 1.0,
                    emitted: true,
                    cap_hit: false,
                }],
            },
        }
    }

    #[test]
    fn transcribe_observation_order_codecs_and_terminal_bytes() {
        let observed = transcribe_responses("r", output(), true, 8192).unwrap();
        let quiet = transcribe_responses("r", output(), false, 8192).unwrap();
        assert_eq!(observed.len(), 3);
        assert_eq!(quiet.len(), 1);
        assert!(matches!(
            observed[0].0,
            WorkerResponse::EncoderOutput { frames: 1, .. }
        ));
        assert!(matches!(
            observed[1].0,
            WorkerResponse::DecodeTrace { steps: 1, .. }
        ));
        assert_eq!(observed[2], quiet[0]);
        assert_eq!(observed[0].1.as_ref().unwrap().len(), 4096);
        assert_eq!(observed[1].1.as_ref().unwrap().len(), 40);
        assert_eq!(
            quiet[0].1.as_ref().unwrap(),
            &synapse_core::encode_i32_frame(&[7, 0, 2])
        );
        assert!(transcribe_responses("r", output(), true, 4095).is_err());
    }

    #[test]
    fn malformed_transcribe_then_ping_keeps_framing() {
        let mut input = std::io::Cursor::new(Vec::new());
        let request = WorkerRequest::Transcribe {
            req_id: "bad".into(),
            model_ref: "absent".into(),
            mel_frames: 2,
            valid_mel_frames: 1,
            observe: false,
        };
        write_json_frame(&mut input, &request, 8192).unwrap();
        write_frame(&mut input, &[0; 512], 8192).unwrap();
        write_json_frame(
            &mut input,
            &WorkerRequest::Ping {
                req_id: "ping".into(),
            },
            8192,
        )
        .unwrap();
        let mut stream = Stream {
            input: std::io::Cursor::new(input.into_inner()),
            output: vec![],
        };
        request_loop(&mut stream, 8192, || Ok(())).unwrap();
        let mut output = std::io::Cursor::new(stream.output);
        let error: WorkerResponse = read_json_frame(&mut output, 8192).unwrap();
        assert!(
            matches!(error, WorkerResponse::Err { code, .. } if code == "invalid_mel_frame_length")
        );
        let pong: WorkerResponse = read_json_frame(&mut output, 8192).unwrap();
        assert!(matches!(pong, WorkerResponse::Pong { req_id, .. } if req_id == "ping"));
    }
}
#[cfg(test)]
fn packed_sequences(
    model: &Model,
    program: &crate::multirow::tight::TightProgram,
    sizes: &[usize],
    passes: &[Vec<usize>],
    raw: &[u8],
) -> Result<Vec<f32>> {
    packed_sequences_progress(model, program, sizes, passes, raw, &mut || Ok(()))
}

fn packed_sequences_progress(
    model: &Model,
    program: &crate::multirow::tight::TightProgram,
    sizes: &[usize],
    passes: &[Vec<usize>],
    raw: &[u8],
    progress: &mut impl FnMut() -> Result<()>,
) -> Result<Vec<f32>> {
    synapse_core::validate_packed_passes(sizes, passes)?;
    let tokens = decode_i32_frame(raw)?;
    let total = sizes
        .iter()
        .try_fold(0usize, |n, &v| n.checked_add(v))
        .context("invalid_request")?;
    anyhow::ensure!(total == tokens.len(), "invalid_request");
    let tokens = tokens
        .into_iter()
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let hidden = model.profile.n("hidden_size");
    let vocabulary = model
        .tensor(&format!("{}embed_tokens.weight", model.profile.prefix()))?
        .len()
        / hidden;
    // Check every token before running the first pass, not while gathering a later pass.
    anyhow::ensure!(
        tokens.iter().all(|&id| (id as usize) < vocabulary),
        "invalid_token"
    );
    let mut offset = 0;
    let rows: Vec<_> = sizes
        .iter()
        .map(|&size| {
            let row = &tokens[offset..offset + size];
            offset += size;
            row
        })
        .collect();
    let mut output = vec![0.0; hidden * rows.len()];
    for pass in passes {
        let vectors = program.run_progress(
            model,
            &pass.iter().map(|&i| rows[i]).collect::<Vec<_>>(),
            progress,
        )?;
        for (&index, vector) in pass.iter().zip(vectors) {
            output[index * hidden..(index + 1) * hidden].copy_from_slice(&vector);
        }
        progress()?;
    }
    Ok(output)
}
fn sequences(
    model: &Model,
    sizes: &[usize],
    raw: &[u8],
    progress: &mut impl FnMut() -> Result<()>,
) -> Result<Vec<f32>> {
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
        let vector = model.run_progress(&tokens[offset..offset + size], progress)?;
        progress()?;
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

    /// Run the signed development test binary with `ANE_TEST_PACKAGES` pointing
    /// at checksum-pinned converted packages. The global Neural Engine process
    /// lock prevents these test programs from overlapping another ANE workload.
    #[test]
    #[ignore = "requires macOS ANE hardware and the pinned Qwen embedding package"]
    fn packed_fixture_v1_is_bit_identical_to_standalone_and_rotary_constants() {
        let lock_path = PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join("Library/Caches/ck-synapse/ane-direct.lock");
        std::fs::create_dir_all(lock_path.parent().unwrap()).unwrap();
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .unwrap();
        lock.try_lock()
            .expect("another process holds the global ANE lane lock");
        private_api().unwrap();
        crate::multirow::tight::tests::runtime_rope_coefficients_match_actual_graph_constant_bits();
        let profile = Profile::select("qwen3-embedding-0.6b.ane-direct-worker", "embed").unwrap();
        let digest = profile.numeric["converted_package_digest"]
            .as_str()
            .unwrap()
            .to_owned();
        let path = PathBuf::from(std::env::var_os("ANE_TEST_PACKAGES").expect("ANE_TEST_PACKAGES"))
            .join("qwen3-embedding-0.6b.safetensors");
        let mut model = Model::load(profile, &path, &digest).unwrap();
        let fixtures = synapse_core::packed_qualification_fixtures_v1();
        let mut references = Vec::new();
        for fixture in &fixtures {
            let mut vectors = Vec::new();
            for row in fixture {
                model
                    .admit(
                        crate::backend::rung(row.len()).unwrap(),
                        &os_build().unwrap(),
                    )
                    .unwrap();
                vectors.push(model.run(row).unwrap());
            }
            references.push(vectors);
        }
        model.evict(512);
        let program = model
            .compile_tight_packing(synapse_core::PACKED_WIDTH)
            .unwrap();
        assert_eq!(program.executable_count(), 28);
        for (fixture, reference) in fixtures.iter().zip(references) {
            let lengths = fixture.iter().map(Vec::len).collect::<Vec<_>>();
            let plan = synapse_core::plan_packed(&lengths).unwrap();
            let mut indices = plan.passes.iter().flatten().copied().collect::<Vec<_>>();
            indices.sort_unstable();
            let passes = plan
                .passes
                .iter()
                .map(|pass| {
                    pass.iter()
                        .map(|i| indices.binary_search(i).unwrap())
                        .collect()
                })
                .collect::<Vec<Vec<usize>>>();
            let ids = indices
                .iter()
                .flat_map(|&i| fixture[i].iter().map(|&id| id as i32))
                .collect::<Vec<_>>();
            let sizes = indices.iter().map(|&i| lengths[i]).collect::<Vec<_>>();
            let raw = synapse_core::encode_i32_frame(&ids);
            let vectors = packed_sequences(&model, &program, &sizes, &passes, &raw).unwrap();
            for (&index, vector) in indices
                .iter()
                .zip(vectors.chunks_exact(model.profile.n("hidden_size")))
            {
                assert!(
                    vector
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(reference[index].iter().map(|v| v.to_bits())),
                    "qualification-v1 packed row {index}, length {}",
                    lengths[index]
                );
            }
            for index in plan.singles {
                if lengths[index] > 256 {
                    model.evict(256);
                }
                model
                    .admit(
                        crate::backend::rung(lengths[index]).unwrap(),
                        &os_build().unwrap(),
                    )
                    .unwrap();
                let vector = model.run(&fixture[index]).unwrap();
                assert!(
                    vector
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(reference[index].iter().map(|v| v.to_bits())),
                    "qualification-v1 standalone fallback row {index}"
                );
            }
        }
    }
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
            // Generous on purpose: a watcher that misses owner death never exits,
            // while a tight real-time bound would fail a correct watcher on a
            // loaded machine.
            assert!(
                disconnected.elapsed() < Duration::from_secs(10),
                "owner death did not interrupt busy worker"
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        eprintln!("supervisor owner-death exit: {:?}", disconnected.elapsed());
        assert!(status.success(), "watcher helper failed: {status}");
        std::fs::remove_file(path).unwrap();
    }
}
