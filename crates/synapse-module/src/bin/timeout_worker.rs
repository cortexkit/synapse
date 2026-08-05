#![forbid(unsafe_code)]

#[cfg(unix)]
fn main() -> anyhow::Result<()> {
    use std::io;
    use std::os::unix::net::UnixStream;
    use std::path::PathBuf;
    use std::thread;
    use std::time::Duration;

    use anyhow::{bail, Context, Result};
    use synapse_core::worker_framing_sync::{
        read_frame, read_json_frame, write_frame, write_json_frame,
    };
    use synapse_core::{
        encode_f32_frame, EngineIdentity, WorkerHello, WorkerHelloAck, WorkerRequest,
        WorkerResponse, DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
    };

    fn argument(name: &str) -> Result<String> {
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == name {
                return args
                    .next()
                    .with_context(|| format!("{name} requires a value"));
            }
        }
        bail!("missing required argument {name}")
    }

    fn optional_milliseconds(name: &str) -> Result<u64> {
        match argument(name) {
            Ok(value) => value.parse().with_context(|| format!("invalid {name}")),
            Err(_) => Ok(0),
        }
    }

    fn run() -> Result<()> {
        let socket = PathBuf::from(argument("--socket")?);
        let nonce = argument("--nonce")?;
        let load_sleep_ms = optional_milliseconds("--load-sleep-ms")?;
        let embed_sleep_ms = optional_milliseconds("--embed-sleep-ms")?;
        let embed_dims = std::env::var("SYNAPSE_TIMEOUT_WORKER_EMBED_DIMS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let embed_n = std::env::var("SYNAPSE_TIMEOUT_WORKER_EMBED_N")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let abort_on_embed = std::env::var("SYNAPSE_TIMEOUT_WORKER_ABORT_ON_EMBED")
            .ok()
            .is_some_and(|value| value == "1");
        let mut stream = UnixStream::connect(&socket)
            .with_context(|| format!("connect mock worker socket {}", socket.display()))?;
        let hello = WorkerHello {
            v: WORKER_PROTOCOL_VERSION,
            nonce,
            engine: EngineIdentity {
                engine: "timeout-mock".to_string(),
                version: "test".to_string(),
                build_flags: Default::default(),
            },
            pid: std::process::id(),
            max_frame: DEFAULT_MAX_FRAME_BYTES,
        };
        write_json_frame(&mut stream, &hello, DEFAULT_MAX_FRAME_BYTES)?;
        let ack: WorkerHelloAck = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES)?;
        if !ack.accept {
            bail!("mock worker handshake rejected");
        }
        let max_frame = ack.max_frame.min(DEFAULT_MAX_FRAME_BYTES);

        loop {
            let frame = match read_frame(&mut stream, max_frame) {
                Ok(frame) => frame,
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(error) => return Err(error).context("read mock worker request"),
            };
            let request: WorkerRequest =
                serde_json::from_slice(&frame).context("decode request")?;
            match request {
                WorkerRequest::Load { req_id, .. } => {
                    thread::sleep(Duration::from_millis(load_sleep_ms));
                    write_json_frame(
                        &mut stream,
                        &WorkerResponse::Loaded {
                            req_id,
                            model_ref: "mock-model-0".to_string(),
                            dims: 1,
                            cold_load_ms: load_sleep_ms,
                        },
                        max_frame,
                    )?;
                }
                WorkerRequest::EmbedBatch { req_id, .. } => {
                    let _ = read_frame(&mut stream, max_frame).context("read embed ids")?;
                    if abort_on_embed {
                        return Ok(());
                    }
                    thread::sleep(Duration::from_millis(embed_sleep_ms));
                    write_json_frame(
                        &mut stream,
                        &WorkerResponse::Vectors {
                            req_id,
                            dims: embed_dims,
                            n: embed_n,
                        },
                        max_frame,
                    )?;
                    let values = vec![1.0_f32; embed_dims.saturating_mul(embed_n)];
                    write_frame(&mut stream, &encode_f32_frame(&values), max_frame)?;
                }
                WorkerRequest::Generate {
                    req_id, max_tokens, ..
                } => {
                    let prompt = read_frame(&mut stream, max_frame).context("read generate ids")?;
                    let generated_token_ids = (max_tokens > 0).then_some(2).into_iter().collect();
                    write_json_frame(
                        &mut stream,
                        &WorkerResponse::Text {
                            req_id,
                            text: "fallback".to_string(),
                            n_prompt: prompt.len() / std::mem::size_of::<i32>(),
                            n_gen: usize::from(max_tokens > 0),
                            finish_reason: "stop".to_string(),
                            generated_token_ids,
                        },
                        max_frame,
                    )?;
                }
                WorkerRequest::Shutdown {} => return Ok(()),
                WorkerRequest::Ping { req_id } => {
                    write_json_frame(
                        &mut stream,
                        &WorkerResponse::Pong {
                            req_id,
                            rss_mb: 0,
                            models_loaded: 1,
                            placement_share: None,
                        },
                        max_frame,
                    )?;
                }
                WorkerRequest::Unload { req_id, .. } => {
                    write_json_frame(&mut stream, &WorkerResponse::Unloaded { req_id }, max_frame)?;
                }
                other => {
                    let req_id = match other {
                        WorkerRequest::Rerank { req_id, .. } => Some(req_id),
                        WorkerRequest::Load { .. }
                        | WorkerRequest::EmbedBatch { .. }
                        | WorkerRequest::Unload { .. }
                        | WorkerRequest::Ping { .. }
                        | WorkerRequest::Generate { .. }
                        | WorkerRequest::Shutdown {} => None,
                    };
                    write_json_frame(
                        &mut stream,
                        &WorkerResponse::Err {
                            req_id,
                            code: "unsupported".to_string(),
                            msg: "timeout mock only supports LOAD and EMBED_BATCH".to_string(),
                        },
                        max_frame,
                    )?;
                }
            }
        }
    }

    run()
}

#[cfg(not(unix))]
fn main() {
    eprintln!("timeout mock worker requires Unix sockets");
}
