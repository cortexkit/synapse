#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, Read, Write};
use std::num::NonZeroU32;
#[cfg(unix)]
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
    sampling::LlamaSampler,
    token::LlamaToken,
    TokenToStringError,
};
use llama_cpp_sys_2::{
    llama_flash_attn_type, LLAMA_FLASH_ATTN_TYPE_AUTO, LLAMA_FLASH_ATTN_TYPE_DISABLED,
    LLAMA_FLASH_ATTN_TYPE_ENABLED,
};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use synapse_core::WorkerHelloAck;
use synapse_core::{
    decode_i32_frame, encode_f32_frame, EngineIdentity, WorkerHello, WorkerPooling, WorkerRequest,
    WorkerResponse, DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
};

const ENGINE_VERSION: &str = "llama-cpp-2-0.1.151";
const MAX_BATCH_SEQUENCES: usize = 256;
const DEFAULT_MAX_GENERATE_TOKENS: u32 = 512;
const LLAMA_TOKEN_NULL: i32 = -1;

#[derive(Parser)]
#[command(name = "ck-synapse-worker-llama")]
struct Args {
    #[arg(long)]
    socket: Option<PathBuf>,
    #[cfg(windows)]
    #[arg(long)]
    pipe: Option<String>,
    #[arg(long)]
    nonce: String,
    #[arg(long = "test-abort", hide = true)]
    test_abort: bool,
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
    max_generate_tokens: u32,
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
            max_generate_tokens: parse_u32(
                values,
                "microllm_max_tokens",
                DEFAULT_MAX_GENERATE_TOKENS,
            )?,
        })
    }
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let hello = WorkerHello {
        v: WORKER_PROTOCOL_VERSION,
        nonce: args.nonce.clone(),
        engine: engine_identity(),
        pid: std::process::id(),
        max_frame: DEFAULT_MAX_FRAME_BYTES,
    };
    run_worker_loop(args, hello)?;
    Ok(())
}

#[cfg(unix)]
fn run_worker_loop(args: Args, hello: WorkerHello) -> Result<u32> {
    let socket = args
        .socket
        .as_ref()
        .context("worker requires --socket on unix")?;
    let mut stream = UnixStream::connect(socket)
        .with_context(|| format!("connect worker socket {}", socket.display()))?;
    synapse_core::worker_framing_sync::write_json_frame(
        &mut stream,
        &hello,
        DEFAULT_MAX_FRAME_BYTES,
    )?;
    let ack: WorkerHelloAck =
        synapse_core::worker_framing_sync::read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES)?;
    ensure!(
        ack.v == WORKER_PROTOCOL_VERSION,
        "module replied with protocol v{}",
        ack.v
    );
    ensure!(ack.accept, "module rejected worker handshake");
    let max_frame = ack.max_frame.min(DEFAULT_MAX_FRAME_BYTES);
    worker_request_loop(&mut stream, max_frame, &args)
}

#[cfg(windows)]
fn run_worker_loop(args: Args, hello: WorkerHello) -> Result<u32> {
    let pipe = args
        .pipe
        .as_deref()
        .context("worker requires --pipe on windows")?;
    let (mut stream, max_frame) =
        synapse_core::worker_transport::windows_client::connect_and_handshake(
            pipe,
            &hello,
            DEFAULT_MAX_FRAME_BYTES,
        )
        .with_context(|| format!("connect worker pipe {pipe}"))?;
    worker_request_loop(&mut stream, max_frame, &args)
}

fn worker_request_loop<S: Read + Write>(
    stream: &mut S,
    max_frame: u32,
    args: &Args,
) -> Result<u32> {
    let mut state = WorkerState::new();
    loop {
        let frame = match synapse_core::worker_framing_sync::read_frame(stream, max_frame) {
            Ok(frame) => frame,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error).context("read request frame"),
        };
        let request: WorkerRequest =
            serde_json::from_slice(&frame).context("decode request JSON")?;
        let should_abort = args.test_abort_on_request
            || (args.test_abort && !matches!(request, WorkerRequest::Load { .. }));
        if should_abort {
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
                synapse_core::worker_framing_sync::write_json_frame(stream, &response, max_frame)?;
            }
            WorkerRequest::EmbedBatch {
                req_id,
                model_ref,
                pooling,
                normalize,
                items,
            } => {
                let raw = synapse_core::worker_framing_sync::read_frame(stream, max_frame)
                    .context("read EMBED_BATCH ids")?;
                let response = match handle_embed_batch(
                    &mut state, &req_id, &model_ref, pooling, normalize, &items, &raw,
                ) {
                    Ok((response, vectors)) => {
                        synapse_core::worker_framing_sync::write_json_frame(
                            stream, &response, max_frame,
                        )?;
                        synapse_core::worker_framing_sync::write_frame(
                            stream, &vectors, max_frame,
                        )?;
                        continue;
                    }
                    Err(error) => WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "inference_failed".to_string(),
                        msg: error.to_string(),
                    },
                };
                synapse_core::worker_framing_sync::write_json_frame(stream, &response, max_frame)?;
            }
            WorkerRequest::Rerank {
                req_id,
                query_n_tokens,
                candidates,
                model_ref,
            } => {
                let raw = synapse_core::worker_framing_sync::read_frame(stream, max_frame)
                    .context("read RERANK ids")?;
                let response = match handle_rerank(
                    &mut state,
                    &req_id,
                    &model_ref,
                    query_n_tokens,
                    &candidates,
                    &raw,
                ) {
                    Ok((response, scores)) => {
                        synapse_core::worker_framing_sync::write_json_frame(
                            stream, &response, max_frame,
                        )?;
                        synapse_core::worker_framing_sync::write_frame(stream, &scores, max_frame)?;
                        continue;
                    }
                    Err(error) => WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "inference_failed".to_string(),
                        msg: error.to_string(),
                    },
                };
                synapse_core::worker_framing_sync::write_json_frame(stream, &response, max_frame)?;
            }
            WorkerRequest::Generate {
                req_id,
                model_ref,
                max_tokens,
                grammar,
            } => {
                let raw = synapse_core::worker_framing_sync::read_frame(stream, max_frame)
                    .context("read GENERATE ids")?;
                let response = match handle_generate(
                    &mut state,
                    &req_id,
                    &model_ref,
                    max_tokens,
                    grammar.as_deref(),
                    &raw,
                ) {
                    Ok(response) => response,
                    Err(error) => WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "inference_failed".to_string(),
                        msg: error.to_string(),
                    },
                };
                synapse_core::worker_framing_sync::write_json_frame(stream, &response, max_frame)?;
            }
            WorkerRequest::Unload { req_id, model_ref } => {
                state.models.remove(&model_ref);
                synapse_core::worker_framing_sync::write_json_frame(
                    stream,
                    &WorkerResponse::Unloaded { req_id },
                    max_frame,
                )?;
            }
            WorkerRequest::Ping { req_id } => {
                synapse_core::worker_framing_sync::write_json_frame(
                    stream,
                    &WorkerResponse::Pong {
                        req_id,
                        rss_mb: 0,
                        models_loaded: state.models.len(),
                        placement_share: None,
                    },
                    max_frame,
                )?;
            }
            WorkerRequest::Shutdown {} => break,
        }
    }
    Ok(max_frame)
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

fn handle_rerank(
    state: &mut WorkerState,
    req_id: &str,
    model_ref: &str,
    query_n_tokens: usize,
    candidates: &[synapse_core::WorkerCandidate],
    raw: &[u8],
) -> Result<(WorkerResponse, Vec<u8>)> {
    ensure!(query_n_tokens > 0, "RERANK query has zero tokens");
    ensure!(
        !candidates.is_empty(),
        "RERANK requires at least one candidate"
    );
    ensure!(
        candidates.len() <= MAX_BATCH_SEQUENCES,
        "too many candidates in one worker request"
    );
    let runtime = state
        .models
        .get(model_ref)
        .ok_or_else(|| anyhow!("unknown model_ref '{model_ref}'"))?;
    reject_qwen3_reranker(runtime)?;

    let ids = decode_i32_frame(raw).map_err(|error| anyhow!(error.to_string()))?;
    let expected_tokens = query_n_tokens
        + candidates
            .iter()
            .map(|candidate| candidate.n_tokens)
            .sum::<usize>();
    ensure!(
        ids.len() == expected_tokens,
        "raw id frame has {} tokens, expected {expected_tokens}",
        ids.len()
    );

    let query = trim_rerank_segment(&runtime.model, &ids[..query_n_tokens]);
    ensure!(
        !query.is_empty(),
        "RERANK query is empty after special-token trimming"
    );
    let mut offset = query_n_tokens;
    let mut sequences = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        ensure!(candidate.n_tokens > 0, "candidate {index} has zero tokens");
        let end = offset + candidate.n_tokens;
        let document = trim_rerank_segment(&runtime.model, &ids[offset..end]);
        ensure!(
            !document.is_empty(),
            "candidate {index} is empty after special-token trimming"
        );
        sequences.push(build_rerank_sequence(&runtime.model, query, document)?);
        offset = end;
    }

    let max_tokens_per_seq = sequences.iter().map(Vec::len).max().unwrap_or(1);
    let mut context = new_context_with(
        runtime,
        LlamaPoolingType::Rank,
        LlamaAttentionType::NonCausal,
        true,
        1,
        max_tokens_per_seq,
        max_tokens_per_seq,
    )?;
    let mut scores = Vec::with_capacity(sequences.len());
    for (seq_id, token_ids) in sequences.iter().enumerate() {
        scores.push(score_rerank_sequence(&mut context, seq_id, token_ids)?);
    }
    Ok((
        WorkerResponse::Scores {
            req_id: req_id.to_string(),
        },
        encode_f32_frame(&scores),
    ))
}

fn score_rerank_sequence(
    context: &mut LlamaContext<'_>,
    seq_id: usize,
    token_ids: &[i32],
) -> Result<f32> {
    context.clear_kv_cache();
    let llama_tokens = token_ids
        .iter()
        .copied()
        .map(LlamaToken::new)
        .collect::<Vec<_>>();
    let mut batch = LlamaBatch::new(token_ids.len(), 1);
    batch
        .add_sequence(&llama_tokens, 0, false)
        .with_context(|| format!("add rerank sequence {seq_id} to llama batch"))?;
    context
        .encode(&mut batch)
        .with_context(|| format!("llama_encode rerank sequence {seq_id} failed"))?;
    let embedding = context
        .embeddings_seq_ith(0)
        .with_context(|| format!("read rerank score for sequence {seq_id}"))?;
    ensure!(
        !embedding.is_empty(),
        "rerank head returned no values for sequence {seq_id}"
    );
    Ok(embedding[0])
}

fn handle_generate(
    state: &mut WorkerState,
    req_id: &str,
    model_ref: &str,
    max_tokens: u32,
    grammar: Option<&str>,
    raw: &[u8],
) -> Result<WorkerResponse> {
    let runtime = state
        .models
        .get(model_ref)
        .ok_or_else(|| anyhow!("unknown model_ref '{model_ref}'"))?;
    let ceiling = runtime.config.max_generate_tokens;
    ensure!(
        max_tokens <= ceiling,
        "max_tokens must be <= configured ceiling {ceiling}"
    );
    let grammar_rule = grammar.filter(|value| !value.trim().is_empty());
    let prompt_ids = decode_i32_frame(raw).map_err(|error| anyhow!(error.to_string()))?;
    ensure!(!prompt_ids.is_empty(), "GENERATE prompt has zero tokens");

    let n_prompt = prompt_ids.len();
    let total_capacity = n_prompt
        .checked_add(usize::try_from(max_tokens).context("max_tokens does not fit usize")?)
        .context("prompt + max_tokens overflowed")?
        .max(1);
    let mut context = new_context_with(
        runtime,
        LlamaPoolingType::None,
        LlamaAttentionType::Causal,
        false,
        1,
        total_capacity,
        total_capacity,
    )?;
    let prompt_tokens = prompt_ids
        .iter()
        .copied()
        .map(LlamaToken::new)
        .collect::<Vec<_>>();
    let mut prompt_batch = LlamaBatch::new(n_prompt, 1);
    prompt_batch
        .add_sequence(&prompt_tokens, 0, false)
        .context("add prompt to llama batch")?;
    context
        .decode(&mut prompt_batch)
        .context("llama_decode prompt failed")?;

    let mut sampler = build_generate_sampler(&runtime.model, grammar_rule)?;
    sampler.accept_many(&prompt_tokens);
    let mut generated = Vec::new();
    let mut finish_reason = "length";
    for step in 0..max_tokens {
        let token = sampler.sample(&context, -1);
        if runtime.model.is_eog_token(token) {
            finish_reason = "stop";
            break;
        }
        sampler.accept(token);
        generated.push(token);
        let pos = i32::try_from(n_prompt)
            .context("prompt length does not fit into i32")?
            .checked_add(i32::try_from(step).context("generation step does not fit into i32")?)
            .context("generation position overflowed")?;
        let mut next_batch = LlamaBatch::new(1, 1);
        next_batch
            .add(token, pos, &[0], true)
            .context("add generated token to llama batch")?;
        context
            .decode(&mut next_batch)
            .context("llama_decode generated token failed")?;
    }
    let text = decode_generated_text(&runtime.model, &generated)?;
    Ok(WorkerResponse::Text {
        req_id: req_id.to_string(),
        text,
        n_prompt,
        n_gen: generated.len(),
        finish_reason: finish_reason.to_string(),
    })
}

fn reject_qwen3_reranker(runtime: &LoadedRuntime) -> Result<()> {
    let architecture = runtime
        .model
        .meta_val_str("general.architecture")
        .unwrap_or_default()
        .to_ascii_lowercase();
    ensure!(
        !architecture.contains("qwen3"),
        "Qwen3 reranker GGUFs are disabled because their rerank templates are not faithful in this worker"
    );
    Ok(())
}

fn build_rerank_sequence(model: &LlamaModel, query: &[i32], document: &[i32]) -> Result<Vec<i32>> {
    let bos = model.token_bos().0;
    let sep = model.token_sep().0;
    let eos = model.token_eos().0;
    ensure!(
        bos != LLAMA_TOKEN_NULL,
        "rerank model vocab has no BOS/CLS token"
    );
    ensure!(
        sep != LLAMA_TOKEN_NULL,
        "rerank model vocab has no SEP token"
    );
    let end = if eos != LLAMA_TOKEN_NULL { eos } else { sep };

    let mut sequence = Vec::with_capacity(query.len() + document.len() + 3);
    sequence.push(bos);
    sequence.extend_from_slice(query);
    sequence.push(sep);
    sequence.extend_from_slice(document);
    sequence.push(end);
    Ok(sequence)
}

fn trim_rerank_segment<'a>(model: &LlamaModel, ids: &'a [i32]) -> &'a [i32] {
    let bos = model.token_bos().0;
    let sep = model.token_sep().0;
    let eos = model.token_eos().0;
    let mut start = 0;
    let mut end = ids.len();
    if ids.first().is_some_and(|token| *token == bos) {
        start = 1;
    }
    while end > start
        && ids
            .get(end - 1)
            .is_some_and(|token| *token == sep || *token == eos)
    {
        end -= 1;
    }
    &ids[start..end]
}

fn decode_generated_text(model: &LlamaModel, tokens: &[LlamaToken]) -> Result<String> {
    let mut bytes = Vec::new();
    for token in tokens {
        bytes.extend(decode_token_piece_bytes(model, *token)?);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn decode_token_piece_bytes(model: &LlamaModel, token: LlamaToken) -> Result<Vec<u8>> {
    match model.token_to_piece_bytes(token, 8, false, None) {
        Ok(bytes) => Ok(bytes),
        Err(TokenToStringError::InsufficientBufferSpace(size)) => model
            .token_to_piece_bytes(
                token,
                usize::try_from(-size).context("token piece size does not fit usize")?,
                false,
                None,
            )
            .context("decode generated token"),
        Err(error) => Err(anyhow!(error)).context("decode generated token"),
    }
}

fn new_context<'a>(
    runtime: &'a LoadedRuntime,
    pooling: WorkerPooling,
    n_seq_max: usize,
    max_tokens_per_seq: usize,
    total_tokens: usize,
) -> Result<LlamaContext<'a>> {
    let forward_pass = runtime.config.forward_pass.resolve(pooling);
    let attention_type = match forward_pass {
        ForwardPass::Encode => LlamaAttentionType::NonCausal,
        ForwardPass::Decode => LlamaAttentionType::Causal,
    };
    new_context_with(
        runtime,
        runtime
            .config
            .pooling_implementation
            .context_pooling_type(pooling),
        attention_type,
        true,
        n_seq_max,
        max_tokens_per_seq,
        total_tokens,
    )
}

fn new_context_with<'a>(
    runtime: &'a LoadedRuntime,
    pooling_type: LlamaPoolingType,
    attention_type: LlamaAttentionType,
    embeddings: bool,
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
    let batch_size = if pooling_type == LlamaPoolingType::Rank {
        total_tokens.max(1)
    } else {
        runtime.config.batch_size.max(total_tokens).max(1)
    };
    let ubatch_size = if pooling_type == LlamaPoolingType::Rank {
        batch_size
    } else {
        runtime.config.ubatch_size.min(batch_size).max(1)
    };
    let n_seq_max = n_seq_max.clamp(1, MAX_BATCH_SEQUENCES);
    let threads = i32::try_from(runtime.config.threads).context("threads does not fit into i32")?;
    let params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_batch(u32::try_from(batch_size).context("batch_size does not fit into u32")?)
        .with_n_ubatch(u32::try_from(ubatch_size).context("ubatch_size does not fit into u32")?)
        .with_n_seq_max(u32::try_from(n_seq_max).context("n_seq_max does not fit into u32")?)
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
        .with_embeddings(embeddings)
        .with_pooling_type(pooling_type)
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

fn parse_u32(values: &BTreeMap<String, String>, key: &str, default: u32) -> Result<u32> {
    values
        .get(key)
        .map(|value| {
            value
                .parse::<u32>()
                .with_context(|| format!("invalid {key} '{value}'"))
        })
        .unwrap_or(Ok(default))
}

fn build_generate_sampler(model: &LlamaModel, grammar: Option<&str>) -> Result<LlamaSampler> {
    if let Some(grammar_str) = grammar {
        let grammar_sampler = LlamaSampler::grammar(model, grammar_str, "root")
            .map_err(|error| anyhow!("grammar sampler init failed: {error}"))?;
        return Ok(LlamaSampler::chain_simple([
            grammar_sampler,
            LlamaSampler::greedy(),
        ]));
    }
    Ok(LlamaSampler::greedy())
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
