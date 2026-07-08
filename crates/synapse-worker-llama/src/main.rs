#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, Read, Write};
use std::net::Shutdown;
use std::num::NonZeroU32;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use clap::Parser;
use llama_cpp_2::{
    context::{
        params::{LlamaAttentionType, LlamaContextParams, LlamaPoolingType},
        LlamaContext,
    },
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, LlamaModel},
    token::LlamaToken,
};
use llama_cpp_sys_2::{
    llama_flash_attn_type, LLAMA_FLASH_ATTN_TYPE_AUTO, LLAMA_FLASH_ATTN_TYPE_DISABLED,
    LLAMA_FLASH_ATTN_TYPE_ENABLED,
};
use sha2::{Digest, Sha256};
use synapse_core::{
    decode_i32_frame, encode_f32_frame, EngineIdentity, WorkerHello, WorkerHelloAck, WorkerPooling,
    WorkerRequest, WorkerResponse, DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
};

const ENGINE_VERSION: &str = "llama-cpp-2-0.1.151";
const MAX_BATCH_SEQUENCES: usize = 256;

#[derive(Parser)]
#[command(name = "synapse-worker-llama")]
struct Args {
    #[arg(long)]
    socket: PathBuf,
    #[arg(long)]
    nonce: String,
    #[arg(long, hide = true)]
    test_abort_on_request: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PoolingImplementation {
    Builtin,
    Manual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlashAttentionSetting {
    Auto,
    Enabled,
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ForwardPassSetting {
    Auto,
    Encode,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ForwardPass {
    Encode,
    Decode,
}

#[derive(Clone, Debug)]
struct WorkerRuntimeConfig {
    ctx_size: usize,
    batch_size: usize,
    ubatch_size: usize,
    gpu_layers: usize,
    threads: usize,
    pooling_implementation: PoolingImplementation,
    flash_attention: FlashAttentionSetting,
    forward_pass: ForwardPassSetting,
}

struct LoadedRuntime {
    backend: LlamaBackend,
    model: LlamaModel,
    config: WorkerRuntimeConfig,
    dims: usize,
}

struct WorkerState {
    models: HashMap<String, LoadedRuntime>,
    next_model: u64,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            models: HashMap::new(),
            next_model: 0,
        }
    }
}

impl PoolingImplementation {
    fn context_pooling_type(self, pooling: WorkerPooling) -> LlamaPoolingType {
        match self {
            Self::Builtin => match pooling {
                WorkerPooling::Mean => LlamaPoolingType::Mean,
                WorkerPooling::Last => LlamaPoolingType::Last,
                WorkerPooling::Cls => LlamaPoolingType::Cls,
            },
            Self::Manual => LlamaPoolingType::None,
        }
    }
}

impl FlashAttentionSetting {
    fn raw_policy(self) -> llama_flash_attn_type {
        match self {
            Self::Auto => LLAMA_FLASH_ATTN_TYPE_AUTO,
            Self::Enabled => LLAMA_FLASH_ATTN_TYPE_ENABLED,
            Self::Disabled => LLAMA_FLASH_ATTN_TYPE_DISABLED,
        }
    }
}

impl ForwardPassSetting {
    fn resolve(self, pooling: WorkerPooling) -> ForwardPass {
        match self {
            Self::Auto => match pooling {
                WorkerPooling::Mean | WorkerPooling::Cls => ForwardPass::Encode,
                WorkerPooling::Last => ForwardPass::Decode,
            },
            Self::Encode => ForwardPass::Encode,
            Self::Decode => ForwardPass::Decode,
        }
    }
}

impl WorkerRuntimeConfig {
    fn from_map(values: &BTreeMap<String, String>) -> Result<Self> {
        Ok(Self {
            ctx_size: parse_usize(values, "ctx_size", 512)?,
            batch_size: parse_usize(values, "batch_size", 4096)?,
            ubatch_size: parse_usize(values, "ubatch_size", 1024)?,
            gpu_layers: parse_usize(values, "gpu_layers", default_gpu_layers())?,
            threads: parse_usize(values, "threads", default_threads())?,
            pooling_implementation: match values
                .get("pooling_implementation")
                .map(String::as_str)
                .unwrap_or("manual")
            {
                "manual" => PoolingImplementation::Manual,
                "builtin" => PoolingImplementation::Builtin,
                other => bail!("invalid pooling_implementation '{other}'"),
            },
            flash_attention: match values
                .get("flash_attention")
                .map(String::as_str)
                .unwrap_or("auto")
            {
                "auto" => FlashAttentionSetting::Auto,
                "enabled" => FlashAttentionSetting::Enabled,
                "disabled" => FlashAttentionSetting::Disabled,
                other => bail!("invalid flash_attention '{other}'"),
            },
            forward_pass: match values
                .get("forward_pass")
                .map(String::as_str)
                .unwrap_or("auto")
            {
                "auto" => ForwardPassSetting::Auto,
                "encode" => ForwardPassSetting::Encode,
                "decode" => ForwardPassSetting::Decode,
                other => bail!("invalid forward_pass '{other}'"),
            },
        })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut stream = UnixStream::connect(&args.socket)
        .with_context(|| format!("connect worker socket {}", args.socket.display()))?;
    let hello = WorkerHello {
        v: WORKER_PROTOCOL_VERSION,
        nonce: args.nonce,
        engine: engine_identity(),
        pid: std::process::id(),
        max_frame: DEFAULT_MAX_FRAME_BYTES,
    };
    write_json_frame(&mut stream, &hello, DEFAULT_MAX_FRAME_BYTES)?;
    let ack: WorkerHelloAck = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES)?;
    ensure!(
        ack.v == WORKER_PROTOCOL_VERSION,
        "module replied with protocol v{}",
        ack.v
    );
    ensure!(ack.accept, "module rejected worker handshake");
    let max_frame = ack.max_frame.min(DEFAULT_MAX_FRAME_BYTES);

    let mut state = WorkerState::new();
    loop {
        let frame = match read_frame(&mut stream, max_frame) {
            Ok(frame) => frame,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error).context("read request frame"),
        };
        let request: WorkerRequest =
            serde_json::from_slice(&frame).context("decode request JSON")?;
        if args.test_abort_on_request {
            std::process::abort();
        }
        match request {
            WorkerRequest::Load {
                req_id,
                artifact_path,
                artifact_digest,
                format,
                runtime_config,
            } => {
                let response = match handle_load(
                    &mut state,
                    req_id.clone(),
                    &artifact_path,
                    &artifact_digest,
                    &format,
                    &runtime_config,
                ) {
                    Ok(response) => response,
                    Err(error) => WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: classify_load_error(&error).to_string(),
                        msg: error.to_string(),
                    },
                };
                write_json_frame(&mut stream, &response, max_frame)?;
            }
            WorkerRequest::EmbedBatch {
                req_id,
                model_ref,
                pooling,
                normalize,
                items,
            } => {
                let raw = read_frame(&mut stream, max_frame).context("read EMBED_BATCH ids")?;
                let response = match handle_embed_batch(
                    &mut state, &req_id, &model_ref, pooling, normalize, &items, &raw,
                ) {
                    Ok((response, vectors)) => {
                        write_json_frame(&mut stream, &response, max_frame)?;
                        write_frame(&mut stream, &vectors, max_frame)?;
                        continue;
                    }
                    Err(error) => WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "inference_failed".to_string(),
                        msg: error.to_string(),
                    },
                };
                write_json_frame(&mut stream, &response, max_frame)?;
            }
            WorkerRequest::Rerank {
                req_id,
                query_n_tokens: _,
                candidates: _,
                model_ref: _,
            } => {
                let _ = read_frame(&mut stream, max_frame).context("read RERANK ids")?;
                write_json_frame(
                    &mut stream,
                    &WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "unknown_type".to_string(),
                        msg: "RERANK is not implemented in the first worker cut".to_string(),
                    },
                    max_frame,
                )?;
            }
            WorkerRequest::Generate {
                req_id,
                model_ref: _,
                max_tokens: _,
                grammar: _,
            } => {
                let _ = read_frame(&mut stream, max_frame).context("read GENERATE ids")?;
                write_json_frame(
                    &mut stream,
                    &WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "unknown_type".to_string(),
                        msg: "GENERATE is not implemented in the first worker cut".to_string(),
                    },
                    max_frame,
                )?;
            }
            WorkerRequest::Unload { req_id, model_ref } => {
                state.models.remove(&model_ref);
                write_json_frame(&mut stream, &WorkerResponse::Unloaded { req_id }, max_frame)?;
            }
            WorkerRequest::Ping { req_id } => {
                write_json_frame(
                    &mut stream,
                    &WorkerResponse::Pong {
                        req_id,
                        rss_mb: 0,
                        models_loaded: state.models.len(),
                    },
                    max_frame,
                )?;
            }
            WorkerRequest::Shutdown {} => {
                let _ = stream.shutdown(Shutdown::Both);
                break;
            }
        }
    }
    Ok(())
}

fn engine_identity() -> EngineIdentity {
    let mut build_flags = BTreeMap::new();
    build_flags.insert("risk_class".to_string(), "abort_capable".to_string());
    build_flags.insert(
        "backend".to_string(),
        if cfg!(target_os = "macos") {
            "metal"
        } else {
            "cpu"
        }
        .to_string(),
    );
    EngineIdentity {
        engine: "llama.cpp".to_string(),
        version: ENGINE_VERSION.to_string(),
        build_flags,
    }
}

fn handle_load(
    state: &mut WorkerState,
    req_id: String,
    artifact_path: &str,
    artifact_digest: &str,
    format: &str,
    runtime_config: &BTreeMap<String, String>,
) -> Result<WorkerResponse> {
    ensure!(
        format == "gguf",
        "llama worker only loads gguf artifacts, got {format}"
    );
    let started = Instant::now();
    let path = Path::new(artifact_path);
    verify_digest(path, artifact_digest)?;
    let config = WorkerRuntimeConfig::from_map(runtime_config)?;
    ensure!(config.ctx_size > 0, "ctx_size must be > 0");
    ensure!(config.batch_size > 0, "batch_size must be > 0");
    ensure!(config.ubatch_size > 0, "ubatch_size must be > 0");
    ensure!(config.threads > 0, "threads must be > 0");

    let backend = LlamaBackend::init().context("llama backend init")?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(
        u32::try_from(config.gpu_layers).context("gpu_layers does not fit into u32")?,
    );
    let model = LlamaModel::load_from_file(&backend, path, &model_params)
        .with_context(|| format!("load gguf model {}", path.display()))?;
    let dims = usize::try_from(model.n_embd()).context("model embedding dims do not fit usize")?;
    let model_ref = format!("llama:{}", state.next_model);
    state.next_model += 1;
    state.models.insert(
        model_ref.clone(),
        LoadedRuntime {
            backend,
            model,
            config,
            dims,
        },
    );
    Ok(WorkerResponse::Loaded {
        req_id,
        model_ref,
        dims,
        cold_load_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
    })
}

fn handle_embed_batch(
    state: &mut WorkerState,
    req_id: &str,
    model_ref: &str,
    pooling: WorkerPooling,
    normalize: bool,
    items: &[synapse_core::WorkerTokenItem],
    raw: &[u8],
) -> Result<(WorkerResponse, Vec<u8>)> {
    ensure!(!items.is_empty(), "EMBED_BATCH requires at least one item");
    ensure!(
        items.len() <= MAX_BATCH_SEQUENCES,
        "too many sequences in one worker request"
    );
    let runtime = state
        .models
        .get(model_ref)
        .ok_or_else(|| anyhow!("unknown model_ref '{model_ref}'"))?;
    let ids = decode_i32_frame(raw).map_err(|error| anyhow!(error.to_string()))?;
    let expected_tokens = items.iter().map(|item| item.n_tokens).sum::<usize>();
    ensure!(
        ids.len() == expected_tokens,
        "raw id frame has {} tokens, expected {expected_tokens}",
        ids.len()
    );
    let mut sequences = Vec::with_capacity(items.len());
    let mut offset = 0_usize;
    for item in items {
        ensure!(item.n_tokens > 0, "item '{}' has zero tokens", item.id);
        let end = offset + item.n_tokens;
        sequences.push(ids[offset..end].to_vec());
        offset = end;
    }

    let max_tokens_per_seq = items.iter().map(|item| item.n_tokens).max().unwrap_or(1);
    let total_tokens = ids.len();
    let mut context = new_context(
        runtime,
        pooling,
        items.len(),
        max_tokens_per_seq,
        total_tokens,
    )?;
    let forward_pass = runtime.config.forward_pass.resolve(pooling);
    let embeddings = embed_token_batches(
        &mut context,
        &sequences,
        pooling,
        runtime.config.pooling_implementation,
        forward_pass,
        normalize,
    )?;
    ensure!(embeddings.len() == items.len(), "embedding count mismatch");
    let dims = embeddings.first().map(Vec::len).unwrap_or(runtime.dims);
    ensure!(
        dims == runtime.dims,
        "embedding dims mismatch: got {dims}, expected {}",
        runtime.dims
    );
    let flat = embeddings.into_iter().flatten().collect::<Vec<_>>();
    Ok((
        WorkerResponse::Vectors {
            req_id: req_id.to_string(),
            dims,
            n: items.len(),
        },
        encode_f32_frame(&flat),
    ))
}

fn new_context<'a>(
    runtime: &'a LoadedRuntime,
    pooling: WorkerPooling,
    n_seq_max: usize,
    max_tokens_per_seq: usize,
    total_tokens: usize,
) -> Result<LlamaContext<'a>> {
    let logical_ctx = runtime.config.ctx_size.max(max_tokens_per_seq).max(1);
    let total_ctx_size = logical_ctx
        .checked_mul(n_seq_max.max(1))
        .context("ctx_size * n_seq_max overflowed")?;
    let ctx_size =
        NonZeroU32::new(u32::try_from(total_ctx_size).context("ctx_size does not fit into u32")?)
            .context("ctx_size must be > 0")?;
    let batch_size = runtime.config.batch_size.max(total_tokens).max(1);
    let ubatch_size = runtime.config.ubatch_size.min(batch_size).max(1);
    let n_seq_max = n_seq_max.clamp(1, MAX_BATCH_SEQUENCES);
    let threads = i32::try_from(runtime.config.threads).context("threads does not fit into i32")?;
    let forward_pass = runtime.config.forward_pass.resolve(pooling);
    let attention_type = match forward_pass {
        ForwardPass::Encode => LlamaAttentionType::NonCausal,
        ForwardPass::Decode => LlamaAttentionType::Causal,
    };
    let params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_batch(u32::try_from(batch_size).context("batch_size does not fit into u32")?)
        .with_n_ubatch(u32::try_from(ubatch_size).context("ubatch_size does not fit into u32")?)
        .with_n_seq_max(u32::try_from(n_seq_max).context("n_seq_max does not fit into u32")?)
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
        .with_embeddings(true)
        .with_pooling_type(
            runtime
                .config
                .pooling_implementation
                .context_pooling_type(pooling),
        )
        .with_attention_type(attention_type)
        .with_flash_attention_policy(runtime.config.flash_attention.raw_policy());
    runtime
        .model
        .new_context(&runtime.backend, params)
        .context("create llama context")
}

fn embed_token_batches(
    context: &mut LlamaContext<'_>,
    sequences: &[Vec<i32>],
    pooling: WorkerPooling,
    pooling_implementation: PoolingImplementation,
    forward_pass: ForwardPass,
    normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    let total_tokens = sequences.iter().map(Vec::len).sum::<usize>();
    ensure!(total_tokens > 0, "cannot embed a batch with zero tokens");
    let seq_count =
        i32::try_from(sequences.len()).context("sequence count does not fit into i32")?;
    let mut batch = LlamaBatch::new(total_tokens, seq_count);
    for (seq_id, token_ids) in sequences.iter().enumerate() {
        ensure!(!token_ids.is_empty(), "sequence {seq_id} has zero tokens");
        let llama_tokens = token_ids
            .iter()
            .copied()
            .map(LlamaToken::new)
            .collect::<Vec<_>>();
        batch
            .add_sequence(
                &llama_tokens,
                i32::try_from(seq_id).context("sequence id does not fit into i32")?,
                true,
            )
            .with_context(|| format!("add sequence {seq_id} to llama batch"))?;
    }
    match forward_pass {
        ForwardPass::Encode => context.encode(&mut batch).context("llama_encode failed")?,
        ForwardPass::Decode => context.decode(&mut batch).context("llama_decode failed")?,
    }

    match pooling_implementation {
        PoolingImplementation::Builtin => collect_builtin_embeddings(context, sequences, normalize),
        PoolingImplementation::Manual => {
            pool_manual_embeddings(context, sequences, pooling, normalize)
        }
    }
}

fn collect_builtin_embeddings(
    context: &LlamaContext<'_>,
    sequences: &[Vec<i32>],
    normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    let mut embeddings = Vec::with_capacity(sequences.len());
    for seq_id in 0..sequences.len() {
        let mut vector = context
            .embeddings_seq_ith(i32::try_from(seq_id).context("sequence id does not fit into i32")?)
            .with_context(|| format!("read sequence embedding {seq_id}"))?
            .to_vec();
        if normalize {
            l2_normalize(&mut vector);
        }
        embeddings.push(vector);
    }
    Ok(embeddings)
}

fn pool_manual_embeddings(
    context: &LlamaContext<'_>,
    sequences: &[Vec<i32>],
    pooling: WorkerPooling,
    normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    let mut pooled = Vec::with_capacity(sequences.len());
    let mut token_offset = 0_i32;
    for sequence in sequences {
        pooled.push(pool_sequence(
            context,
            token_offset,
            sequence.len(),
            pooling,
            normalize,
        )?);
        token_offset +=
            i32::try_from(sequence.len()).context("token count does not fit into i32")?;
    }
    Ok(pooled)
}

fn pool_sequence(
    context: &LlamaContext<'_>,
    start_token: i32,
    token_count: usize,
    pooling: WorkerPooling,
    normalize: bool,
) -> Result<Vec<f32>> {
    ensure!(token_count > 0, "cannot pool an empty sequence");
    let mut vector = match pooling {
        WorkerPooling::Mean => {
            let first = context
                .embeddings_ith(start_token)
                .context("read first token embedding")?;
            let mut pooled = vec![0.0_f32; first.len()];
            for index in 0..token_count {
                let embedding = context
                    .embeddings_ith(
                        start_token
                            + i32::try_from(index).context("token index does not fit into i32")?,
                    )
                    .with_context(|| format!("read token embedding {index}"))?;
                for (slot, value) in pooled.iter_mut().zip(embedding.iter().copied()) {
                    *slot += value;
                }
            }
            let denom = token_count as f32;
            for value in &mut pooled {
                *value /= denom;
            }
            pooled
        }
        WorkerPooling::Last => context
            .embeddings_ith(
                start_token
                    + i32::try_from(token_count - 1)
                        .context("last token index does not fit into i32")?,
            )
            .context("read last token embedding")?
            .to_vec(),
        WorkerPooling::Cls => context
            .embeddings_ith(start_token)
            .context("read cls token embedding")?
            .to_vec(),
    };
    if normalize {
        l2_normalize(&mut vector);
    }
    Ok(vector)
}

fn l2_normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt() + 1e-12;
    for value in vector {
        *value /= norm;
    }
}

fn parse_usize(values: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize> {
    values
        .get(key)
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("invalid {key} '{value}'"))
        })
        .unwrap_or(Ok(default))
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .div_ceil(2)
        .max(1)
}

fn default_gpu_layers() -> usize {
    if cfg!(target_os = "macos") {
        99
    } else {
        0
    }
}

fn classify_load_error(error: &anyhow::Error) -> &'static str {
    let message = error.to_string();
    if message.contains("digest mismatch") || message.contains("only loads gguf") {
        "artifact_invalid"
    } else if message.contains("invalid") || message.contains("must be") {
        "config_invalid"
    } else {
        "artifact_invalid"
    }
}

fn verify_digest(path: &Path, digest: &str) -> Result<()> {
    if digest.trim().is_empty() {
        return Ok(());
    }
    let expected = digest.strip_prefix("sha256:").unwrap_or(digest);
    let actual = sha256_hex(path).with_context(|| format!("hash {}", path.display()))?;
    ensure!(
        actual == expected,
        "digest mismatch for {}: expected {expected}, got {actual}",
        path.display()
    );
    Ok(())
}

fn sha256_hex(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn read_json_frame<T: serde::de::DeserializeOwned>(
    stream: &mut UnixStream,
    max_frame: u32,
) -> Result<T> {
    let frame = read_frame(stream, max_frame)?;
    serde_json::from_slice(&frame).context("decode JSON frame")
}

fn write_json_frame<T: serde::Serialize>(
    stream: &mut UnixStream,
    value: &T,
    max_frame: u32,
) -> Result<()> {
    let bytes = serde_json::to_vec(value).context("encode JSON frame")?;
    write_frame(stream, &bytes, max_frame)
}

fn read_frame(stream: &mut UnixStream, max_frame: u32) -> io::Result<Vec<u8>> {
    let mut len_bytes = [0_u8; 4];
    stream.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes);
    if len > max_frame {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {max_frame}"),
        ));
    }
    let mut frame = vec![0_u8; len as usize];
    stream.read_exact(&mut frame)?;
    Ok(frame)
}

fn write_frame(stream: &mut UnixStream, bytes: &[u8], max_frame: u32) -> Result<()> {
    let len = u32::try_from(bytes.len()).context("frame too large for u32 length")?;
    ensure!(
        len <= max_frame,
        "frame length {len} exceeds max {max_frame}"
    );
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()?;
    Ok(())
}
