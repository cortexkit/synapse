use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{self, Read},
    net::Shutdown,
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use owned_decode_worker::{
    error::DecodeError,
    identity::{CONSTRAINT_ENCODING_ID, WORKER_PROTOCOL_ID},
    protocol::{
        DecodeTransportRequest, DecodeTransportResponse, FinalResponse, FinishReason,
        FrameEnvelope, GenerateContinue, GenerateProgress, GenerateStart, TokenIdJsonConstraint,
        WorkerFrame,
    },
    validation::{validate_start, WorkerStartContext},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use synapse_core::{
    worker_framing_sync::{read_frame, write_json_frame},
    worker_protocol::{
        WorkerHello, WorkerHelloAck, WorkerRequest, WorkerResponse, DEFAULT_MAX_FRAME_BYTES,
        WORKER_PROTOCOL_VERSION,
    },
    EngineIdentity, Fingerprint,
};
use synapse_engine_owned::{
    owned_decode_engine::{
        top_logits, DecodeKernel, Lfm2DecodeModel, Lfm2HybridStepCache, Lfm2HybridStepEngine,
        MetalStepDecoder, MetalStepKvCache, Qwen3DecodeModel, TokenVocabulary, WeightQuantization,
    },
    Precision,
};
use synapse_module::{
    owned_decode_grammar_scheduler::{
        grammar_automaton::{Automaton, State},
        grammar_compile::{TokenIdJsonConstraintV1, INITIAL_STATE_ENCODING},
        grammar_limits::REPRESENTATION_REVISION,
        load_automaton, GrammarSubsetManifest,
    },
    owned_decode_routing::identity::{ConstraintFingerprintInputs, ConstraintRuntimeIdentity},
};
use tokenizers::Tokenizer;

const PRODUCTION_N: u32 = 16;
const ENGINE_VERSION: &str = "owned-metal-decode-v1";

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    socket: PathBuf,
    #[arg(long)]
    nonce: String,
    #[arg(long, hide = true)]
    test_abort_on_request: bool,
}

enum DecodeEngine {
    Qwen { decoder: MetalStepDecoder<'static> },
    Lfm2 { engine: Lfm2HybridStepEngine },
}

enum DecodeCache {
    Qwen(MetalStepKvCache),
    Lfm2(Lfm2HybridStepCache),
}

impl DecodeEngine {
    fn reset(&mut self) -> Result<()> {
        if let Self::Lfm2 { engine } = self {
            engine.reset()?;
        }
        Ok(())
    }

    fn prefill_greedy(&mut self, prompt: &[u32]) -> Result<(DecodeCache, u32)> {
        match self {
            Self::Qwen { decoder } => {
                let (cache, token) = decoder.prefill(prompt)?;
                Ok((DecodeCache::Qwen(cache), token))
            }
            Self::Lfm2 { engine } => {
                let (cache, token) = DecodeKernel::prefill(engine, prompt)?;
                Ok((DecodeCache::Lfm2(cache), token))
            }
        }
    }

    fn prefill_logits(&mut self, prompt: &[u32]) -> Result<(DecodeCache, Vec<f32>)> {
        ensure!(!prompt.is_empty(), "decode prompt must not be empty");
        match self {
            Self::Qwen { decoder } => {
                let mut cache = MetalStepKvCache { position: 0 };
                let mut logits = Vec::new();
                for &token in prompt {
                    logits = decoder.advance(&mut cache, token)?;
                }
                Ok((DecodeCache::Qwen(cache), logits))
            }
            Self::Lfm2 { engine } => {
                let mut cache = Lfm2HybridStepCache { position: 0 };
                let mut logits = Vec::new();
                for &token in prompt {
                    logits = engine.advance(&mut cache, token)?;
                }
                Ok((DecodeCache::Lfm2(cache), logits))
            }
        }
    }

    fn advance(&mut self, cache: &mut DecodeCache, token: u32) -> Result<Vec<f32>> {
        match (self, cache) {
            (Self::Qwen { decoder }, DecodeCache::Qwen(cache)) => decoder.advance(cache, token),
            (Self::Lfm2 { engine }, DecodeCache::Lfm2(cache)) => engine.advance(cache, token),
            _ => bail!("owned decode engine/cache family mismatch"),
        }
    }
}

struct LoadedRuntime {
    model_ref: String,
    decode_fingerprint: String,
    runtime_config_digest: String,
    production_n: u32,
    stop_ids: Vec<u32>,
    vocabulary: Arc<TokenVocabulary>,
    vocabulary_digest: String,
    engine: DecodeEngine,
}

struct ActiveConstraint {
    automaton: Automaton,
    state: State,
    identity: String,
}

struct ResidentGeneration {
    generation_id: String,
    max_tokens: u32,
    generated_ids: Vec<u32>,
    quantum_sequence: u32,
    cache: DecodeCache,
    next_logits: Option<Vec<f32>>,
    next_greedy: Option<u32>,
    constraint: Option<ActiveConstraint>,
}

struct WorkerState {
    worker_generation: u64,
    loaded: Option<LoadedRuntime>,
    resident: Option<ResidentGeneration>,
}

impl WorkerState {
    fn new(worker_generation: u64) -> Self {
        Self {
            worker_generation,
            loaded: None,
            resident: None,
        }
    }

    fn load(
        &mut self,
        req_id: String,
        artifact_path: &str,
        artifact_digest: &str,
        format: &str,
        runtime_config: &BTreeMap<String, String>,
    ) -> Result<WorkerResponse> {
        ensure!(
            self.resident.is_none(),
            "cannot load during an active generation"
        );
        ensure!(
            self.loaded.is_none(),
            "owned decode worker hosts one model key"
        );
        ensure!(
            matches!(format, "safetensors" | "owned-safetensors" | "q8_0"),
            "owned decode worker cannot load format {format}"
        );
        let started = Instant::now();
        let path = Path::new(artifact_path);
        verify_digest(path, artifact_digest)?;
        let family = required_config(runtime_config, "family")?;
        let quant = match required_config(runtime_config, "weight_quant")? {
            "f16" => WeightQuantization::None,
            "q8_0" => WeightQuantization::Q8_0,
            other => bail!("unsupported owned decode weight quantization {other}"),
        };
        let bucket = required_config(runtime_config, "context_bucket")?
            .parse::<usize>()
            .context("parse owned decode context_bucket")?;
        ensure!(
            [512, 1024, 2048].contains(&bucket),
            "unsupported context bucket"
        );
        let production_n = required_config(runtime_config, "production_n")?
            .parse::<u32>()
            .context("parse owned decode production_n")?;
        ensure!(
            production_n == PRODUCTION_N,
            "worker requires committed N=16"
        );

        let tokenizer_path = runtime_config
            .get("tokenizer_path")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_root(path).join("tokenizer.json"));
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            anyhow::anyhow!("load tokenizer {}: {error}", tokenizer_path.display())
        })?;
        let vocabulary = Arc::new(TokenVocabulary::from_tokenizer(&tokenizer)?);
        let vocabulary_digest = token_vocabulary_digest(&vocabulary);

        let (engine, stop_ids) = match family {
            "qwen3-0.6b" => {
                let model = Box::leak(Box::new(Qwen3DecodeModel::load_with_quant(
                    path,
                    Precision::F16,
                    quant,
                )?));
                let stop_ids = model.generation_stop_ids().to_vec();
                let decoder = MetalStepDecoder::new(model, Precision::F16, bucket, quant)?;
                (DecodeEngine::Qwen { decoder }, stop_ids)
            }
            "lfm2-1.2b" => {
                let model = Lfm2DecodeModel::load_with_quant(path, Precision::F16, quant)?;
                let stop_ids = model.generation_stop_ids().to_vec();
                let engine = Lfm2HybridStepEngine::new(&model, Precision::F16, bucket, quant)?;
                (DecodeEngine::Lfm2 { engine }, stop_ids)
            }
            other => bail!("unsupported owned decode family {other}"),
        };
        let model_ref = "owned-decode:0".to_string();
        self.loaded = Some(LoadedRuntime {
            model_ref: model_ref.clone(),
            decode_fingerprint: required_config(runtime_config, "decode_fingerprint")?.to_string(),
            runtime_config_digest: required_config(runtime_config, "runtime_config_digest")?
                .to_string(),
            production_n,
            stop_ids,
            vocabulary,
            vocabulary_digest,
            engine,
        });
        Ok(WorkerResponse::Loaded {
            req_id,
            model_ref,
            dims: 0,
            cold_load_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        })
    }

    fn start(&mut self, start: GenerateStart) -> FrameEnvelope {
        match self.try_start(start) {
            Ok(frame) => FrameEnvelope::new(frame),
            Err(error) => error_frame(error),
        }
    }

    fn try_start(&mut self, start: GenerateStart) -> Result<WorkerFrame, DecodeError> {
        if self.resident.is_some() {
            return Err(DecodeError::ProtocolMismatch);
        }
        let loaded = self
            .loaded
            .as_mut()
            .ok_or(DecodeError::RuntimeConfigMismatch)?;
        let active_constraint = match start.constraint.as_ref() {
            Some(constraint) => Some(load_constraint(
                constraint,
                &start.decode_fingerprint,
                &loaded.vocabulary_digest,
            )?),
            None => None,
        };
        let context = WorkerStartContext {
            loaded_model_ref: loaded.model_ref.clone(),
            decode_fingerprint: loaded.decode_fingerprint.clone(),
            runtime_config_digest: loaded.runtime_config_digest.clone(),
            expected_constraint: start.constraint.clone(),
        };
        let authorization = validate_start(&start, &context, loaded.production_n)?;
        loaded
            .engine
            .reset()
            .map_err(|_| DecodeError::Unavailable)?;
        let (cache, next_logits, next_greedy) = if active_constraint.is_some() {
            let (cache, logits) = loaded
                .engine
                .prefill_logits(&start.prompt_ids)
                .map_err(|_| DecodeError::Unavailable)?;
            (cache, Some(logits), None)
        } else {
            let (cache, token) = loaded
                .engine
                .prefill_greedy(&start.prompt_ids)
                .map_err(|_| DecodeError::Unavailable)?;
            (cache, None, Some(token))
        };
        self.resident = Some(ResidentGeneration {
            generation_id: start.generation_id,
            max_tokens: start.max_tokens,
            generated_ids: Vec::with_capacity(start.max_tokens as usize),
            quantum_sequence: 0,
            cache,
            next_logits,
            next_greedy,
            constraint: active_constraint,
        });
        self.run_quantum(authorization.first_quantum_budget)
    }

    fn continue_generation(&mut self, continuation: GenerateContinue) -> FrameEnvelope {
        match self.try_continue(continuation) {
            Ok(frame) => FrameEnvelope::new(frame),
            Err(error) => error_frame(error),
        }
    }

    fn try_continue(&mut self, continuation: GenerateContinue) -> Result<WorkerFrame, DecodeError> {
        let resident = self
            .resident
            .as_ref()
            .ok_or(DecodeError::ProtocolMismatch)?;
        let remaining = resident
            .max_tokens
            .saturating_sub(resident.generated_ids.len() as u32);
        if continuation.generation_id != resident.generation_id
            || continuation.next_expected_sequence != resident.quantum_sequence.saturating_add(1)
            || continuation.next_token_budget == 0
            || continuation.next_token_budget > PRODUCTION_N
            || continuation.next_token_budget > remaining
        {
            return Err(DecodeError::ProtocolMismatch);
        }
        self.run_quantum(continuation.next_token_budget)
    }

    fn run_quantum(&mut self, token_budget: u32) -> Result<WorkerFrame, DecodeError> {
        for _ in 0..token_budget {
            let loaded = self
                .loaded
                .as_mut()
                .ok_or(DecodeError::RuntimeConfigMismatch)?;
            let resident = self
                .resident
                .as_mut()
                .ok_or(DecodeError::ProtocolMismatch)?;
            let token = if let Some(constraint) = resident.constraint.as_mut() {
                let logits = resident
                    .next_logits
                    .take()
                    .ok_or(DecodeError::ProtocolMismatch)?;
                constrained_top1(&logits, &loaded.stop_ids, &loaded.vocabulary, constraint)?
            } else {
                resident
                    .next_greedy
                    .take()
                    .ok_or(DecodeError::ProtocolMismatch)?
            };

            if loaded.stop_ids.contains(&token) {
                if resident
                    .constraint
                    .as_ref()
                    .is_some_and(|constraint| !constraint.automaton.is_complete(&constraint.state))
                {
                    self.resident = None;
                    return Err(DecodeError::GrammarStopBeforeCompletion);
                }
                return Ok(self.finish(FinishReason::StopToken));
            }

            if let Some(constraint) = resident.constraint.as_mut() {
                let piece = loaded
                    .vocabulary
                    .token_piece(token)
                    .ok_or(DecodeError::GrammarUnsatisfiable)?;
                constraint.state = constraint
                    .automaton
                    .commit_token(&constraint.state, piece)
                    .map_err(|_| DecodeError::GrammarUnsatisfiable)?;
            }
            resident.generated_ids.push(token);

            if resident
                .constraint
                .as_ref()
                .is_some_and(|constraint| constraint.automaton.is_complete(&constraint.state))
            {
                return Ok(self.finish(FinishReason::GrammarComplete));
            }
            if resident.generated_ids.len() as u32 == resident.max_tokens {
                if resident.constraint.is_some() {
                    self.resident = None;
                    return Err(DecodeError::GrammarMaxTokensExhausted);
                }
                return Ok(self.finish(FinishReason::MaxTokens));
            }

            let logits = loaded
                .engine
                .advance(&mut resident.cache, token)
                .map_err(|_| DecodeError::Unavailable)?;
            if resident.constraint.is_some() {
                resident.next_logits = Some(logits);
            } else {
                resident.next_greedy = top_logits(&logits, 1).first().map(|entry| entry.token_id);
            }
        }

        let resident = self
            .resident
            .as_mut()
            .ok_or(DecodeError::ProtocolMismatch)?;
        resident.quantum_sequence = resident.quantum_sequence.saturating_add(1);
        Ok(WorkerFrame::Progress(GenerateProgress {
            generation_id: resident.generation_id.clone(),
            quantum_sequence: resident.quantum_sequence,
            committed_token_count: resident.generated_ids.len() as u32,
        }))
    }

    fn finish(&mut self, finish_reason: FinishReason) -> WorkerFrame {
        let resident = self
            .resident
            .take()
            .expect("finish is called only for a resident generation");
        WorkerFrame::Final(FinalResponse {
            generation_id: resident.generation_id,
            committed_token_count: resident.generated_ids.len() as u32,
            generated_ids: resident.generated_ids,
            decode_fingerprint: self
                .loaded
                .as_ref()
                .expect("generation has a loaded model")
                .decode_fingerprint
                .clone(),
            runtime_config_digest: self
                .loaded
                .as_ref()
                .expect("generation has a loaded model")
                .runtime_config_digest
                .clone(),
            worker_generation: self.worker_generation,
            finish_reason,
            constraint_identity: resident
                .constraint
                .as_ref()
                .map(|constraint| constraint.identity.clone()),
            constraint_complete: resident
                .constraint
                .as_ref()
                .is_some_and(|constraint| constraint.automaton.is_complete(&constraint.state)),
            last_completed_sequence: resident.quantum_sequence,
        })
    }

    fn cancel(
        &mut self,
        cancellation: &owned_decode_worker::protocol::GenerateCancel,
    ) -> Result<u32, DecodeError> {
        let resident = self.resident.take().ok_or(DecodeError::ProtocolMismatch)?;
        if cancellation.generation_id != resident.generation_id {
            self.resident = Some(resident);
            return Err(DecodeError::ProtocolMismatch);
        }
        Ok(resident.generated_ids.len() as u32)
    }
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let worker_generation = worker_generation(&args.nonce)?;
    let mut stream = UnixStream::connect(&args.socket)
        .with_context(|| format!("connect worker socket {}", args.socket.display()))?;
    let hello = WorkerHello {
        v: WORKER_PROTOCOL_VERSION,
        nonce: args.nonce,
        engine: engine_identity(worker_generation),
        pid: std::process::id(),
        max_frame: DEFAULT_MAX_FRAME_BYTES,
    };
    write_json_frame(&mut stream, &hello, DEFAULT_MAX_FRAME_BYTES)?;
    let ack: WorkerHelloAck = read_json(&mut stream, DEFAULT_MAX_FRAME_BYTES)?;
    ensure!(
        ack.v == WORKER_PROTOCOL_VERSION,
        "module replied with wrong protocol"
    );
    ensure!(ack.accept, "module rejected worker handshake");
    let max_frame = ack.max_frame.min(DEFAULT_MAX_FRAME_BYTES);
    let mut state = WorkerState::new(worker_generation);

    loop {
        let bytes = match read_frame(&mut stream, max_frame) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error).context("read worker request"),
        };
        if args.test_abort_on_request {
            std::process::abort();
        }
        let value: Value = serde_json::from_slice(&bytes).context("decode request JSON")?;
        let ty = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if matches!(
            ty,
            "GENERATE_START" | "GENERATE_CONTINUE" | "GENERATE_CANCEL"
        ) {
            let request: DecodeTransportRequest =
                serde_json::from_value(value).context("decode owned worker request")?;
            handle_decode_request(&mut stream, max_frame, &mut state, request)?;
            continue;
        }
        let request: WorkerRequest =
            serde_json::from_value(value).context("decode worker request")?;
        match request {
            WorkerRequest::Load {
                req_id,
                artifact_path,
                artifact_digest,
                format,
                runtime_config,
            } => {
                let response = state
                    .load(
                        req_id.clone(),
                        &artifact_path,
                        &artifact_digest,
                        &format,
                        &runtime_config,
                    )
                    .unwrap_or_else(|error| WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "load_failed".to_string(),
                        msg: error.to_string(),
                    });
                write_json_frame(&mut stream, &response, max_frame)?;
            }
            WorkerRequest::Ping { req_id } => write_json_frame(
                &mut stream,
                &WorkerResponse::Pong {
                    req_id,
                    rss_mb: 0,
                    models_loaded: usize::from(state.loaded.is_some()),
                    placement_share: None,
                },
                max_frame,
            )?,
            WorkerRequest::Unload { req_id, model_ref } => {
                if state.resident.is_some() {
                    write_json_frame(
                        &mut stream,
                        &WorkerResponse::Err {
                            req_id: Some(req_id),
                            code: DecodeError::ProtocolMismatch.as_str().to_string(),
                            msg: "cannot unload an active resident generation".to_string(),
                        },
                        max_frame,
                    )?;
                } else {
                    ensure!(
                        state
                            .loaded
                            .as_ref()
                            .is_some_and(|loaded| loaded.model_ref == model_ref),
                        "unknown owned decode model ref"
                    );
                    state.loaded = None;
                    write_json_frame(&mut stream, &WorkerResponse::Unloaded { req_id }, max_frame)?;
                }
            }
            WorkerRequest::Shutdown {} => {
                state.resident = None;
                state.loaded = None;
                let _ = stream.shutdown(Shutdown::Both);
                break;
            }
            other => {
                let req_id = standard_req_id(&other);
                write_json_frame(
                    &mut stream,
                    &WorkerResponse::Err {
                        req_id,
                        code: "unknown_type".to_string(),
                        msg: "decode worker supports LOAD, owned generation frames, PING, UNLOAD, and SHUTDOWN only".to_string(),
                    },
                    max_frame,
                )?;
            }
        }
    }
    Ok(())
}

fn handle_decode_request(
    stream: &mut UnixStream,
    max_frame: u32,
    state: &mut WorkerState,
    request: DecodeTransportRequest,
) -> Result<()> {
    let response = match request {
        DecodeTransportRequest::GenerateStart { req_id, start } => DecodeTransportResponse::Frame {
            req_id,
            envelope: state.start(*start),
        },
        DecodeTransportRequest::GenerateContinue {
            req_id,
            continuation,
        } => DecodeTransportResponse::Frame {
            req_id,
            envelope: state.continue_generation(continuation),
        },
        DecodeTransportRequest::GenerateCancel {
            req_id,
            cancellation,
        } => match state.cancel(&cancellation) {
            Ok(committed_token_count) => DecodeTransportResponse::Cancelled {
                req_id,
                generation_id: cancellation.generation_id,
                committed_token_count,
            },
            Err(error) => DecodeTransportResponse::Frame {
                req_id,
                envelope: error_frame(error),
            },
        },
    };
    write_json_frame(stream, &response, max_frame)?;
    Ok(())
}

fn load_constraint(
    constraint: &TokenIdJsonConstraint,
    decode_fingerprint: &str,
    vocabulary_digest: &str,
) -> Result<ActiveConstraint, DecodeError> {
    let manifest = GrammarSubsetManifest::default();
    if constraint.encoding_id != CONSTRAINT_ENCODING_ID
        || constraint.encoding_id != REPRESENTATION_REVISION
        || constraint.grammar_subset_revision != manifest.grammar_subset_revision
        || constraint.grammar_compiler_revision != manifest.grammar_compiler_revision
        || constraint.tokenizer_vocabulary_digest != vocabulary_digest
        || constraint.limits_manifest_id != manifest.limits_manifest_id
        || constraint.worker_constraint_runtime_revision
            != manifest.worker_constraint_runtime_revision
        || constraint.initial_state_encoding != INITIAL_STATE_ENCODING
    {
        return Err(DecodeError::ConstraintVersionMismatch);
    }
    let runtime_identity = ConstraintRuntimeIdentity {
        base_decode_fingerprint: Fingerprint(decode_fingerprint.to_string()),
        representation_revision: constraint.encoding_id.clone(),
        grammar_subset_revision: constraint.grammar_subset_revision.clone(),
        grammar_compiler_revision: constraint.grammar_compiler_revision.clone(),
        tokenizer_vocabulary_digest: constraint.tokenizer_vocabulary_digest.clone(),
        limits_manifest_id: constraint.limits_manifest_id.clone(),
        worker_constraint_runtime_revision: constraint.worker_constraint_runtime_revision.clone(),
    };
    if runtime_identity.digest() != constraint.constraint_runtime_identity {
        return Err(DecodeError::ConstraintVersionMismatch);
    }
    let wire = TokenIdJsonConstraintV1 {
        representation_revision: constraint.encoding_id.clone(),
        constraint_runtime_identity: runtime_identity,
        constraint_fingerprint: Fingerprint(constraint.constraint_fingerprint.clone()),
        tokenizer_vocabulary_digest: constraint.tokenizer_vocabulary_digest.clone(),
        limits_manifest_id: constraint.limits_manifest_id.clone(),
        canonical_schema_digest: constraint.canonical_schema_digest.clone(),
        initial_state_encoding: constraint.initial_state_encoding.clone(),
        initial_state_digest: constraint.initial_state_digest.clone(),
        compiled_automaton_digest: constraint.compiled_automaton_digest.clone(),
        automaton_bytes: constraint.automaton_bytes.clone(),
    };
    let automaton =
        load_automaton(&wire, &manifest).map_err(|_| DecodeError::ConstraintVersionMismatch)?;
    let schema_bytes = serde_json::to_vec(automaton.schema())
        .map_err(|_| DecodeError::ConstraintVersionMismatch)?;
    if sha256_hex(&schema_bytes) != constraint.canonical_schema_digest {
        return Err(DecodeError::ConstraintVersionMismatch);
    }
    let root_type = format!("{:?}", automaton.schema().root().ty);
    let initial_bytes = serde_json::to_vec(&serde_json::json!({
        "encoding": INITIAL_STATE_ENCODING,
        "root_type": root_type,
        "stack_depth": 0,
        "complete": false,
    }))
    .map_err(|_| DecodeError::ConstraintVersionMismatch)?;
    if sha256_hex(&initial_bytes) != constraint.initial_state_digest {
        return Err(DecodeError::ConstraintVersionMismatch);
    }
    let fingerprint = ConstraintFingerprintInputs {
        runtime_identity_digest: constraint.constraint_runtime_identity.clone(),
        canonical_schema_digest: constraint.canonical_schema_digest.clone(),
        initial_state_encoding: constraint.initial_state_encoding.clone(),
        initial_state_digest: constraint.initial_state_digest.clone(),
        compiled_automaton_digest: constraint.compiled_automaton_digest.clone(),
    }
    .fingerprint();
    if fingerprint.0 != constraint.constraint_fingerprint {
        return Err(DecodeError::ConstraintVersionMismatch);
    }
    let state = automaton.initial();
    Ok(ActiveConstraint {
        automaton,
        state,
        identity: constraint.constraint_runtime_identity.clone(),
    })
}

fn constrained_top1(
    logits: &[f32],
    stop_ids: &[u32],
    vocabulary: &TokenVocabulary,
    constraint: &ActiveConstraint,
) -> Result<u32, DecodeError> {
    let stops: HashSet<u32> = stop_ids.iter().copied().collect();
    let mut selected: Option<(u32, f32)> = None;
    for (index, &logit) in logits.iter().enumerate() {
        let token_id = index as u32;
        let permitted = stops.contains(&token_id)
            || vocabulary.token_piece(token_id).is_some_and(|piece| {
                constraint
                    .automaton
                    .token_is_permitted(&constraint.state, piece)
            });
        if !permitted {
            continue;
        }
        if selected.is_none_or(|(current_id, current)| {
            logit.total_cmp(&current).is_gt()
                || (logit.total_cmp(&current).is_eq() && token_id < current_id)
        }) {
            selected = Some((token_id, logit));
        }
    }
    selected
        .map(|(token_id, _)| token_id)
        .ok_or(DecodeError::GrammarUnsatisfiable)
}

fn token_vocabulary_digest(vocabulary: &TokenVocabulary) -> String {
    let mut hasher = Sha256::new();
    for token_id in 0..vocabulary.len() {
        if let Some(piece) = vocabulary.token_piece(token_id as u32) {
            hasher.update(piece);
        }
        hasher.update([0]);
    }
    hex::encode(hasher.finalize())
}

fn error_frame(error: DecodeError) -> FrameEnvelope {
    FrameEnvelope::new(WorkerFrame::Error {
        id: error.as_str().to_string(),
    })
}

fn engine_identity(worker_generation: u64) -> EngineIdentity {
    let mut build_flags = BTreeMap::new();
    build_flags.insert("backend".to_string(), "metal".to_string());
    build_flags.insert("lane".to_string(), "decode".to_string());
    build_flags.insert("protocol".to_string(), WORKER_PROTOCOL_ID.to_string());
    build_flags.insert(
        "constraint_encoding".to_string(),
        CONSTRAINT_ENCODING_ID.to_string(),
    );
    build_flags.insert("risk_class".to_string(), "abort_capable".to_string());
    build_flags.insert(
        "worker_generation".to_string(),
        worker_generation.to_string(),
    );
    EngineIdentity {
        engine: "owned-metal-decode".to_string(),
        version: ENGINE_VERSION.to_string(),
        build_flags,
    }
}

fn standard_req_id(request: &WorkerRequest) -> Option<String> {
    match request {
        WorkerRequest::EmbedBatch { req_id, .. }
        | WorkerRequest::Rerank { req_id, .. }
        | WorkerRequest::Generate { req_id, .. } => Some(req_id.clone()),
        _ => None,
    }
}

fn required_config<'a>(config: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str> {
    config
        .get(key)
        .map(String::as_str)
        .with_context(|| format!("owned decode runtime config is missing {key}"))
}

fn model_root(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or(path).to_path_buf()
    }
}

fn worker_generation(nonce: &str) -> Result<u64> {
    ensure!(
        nonce.len() == 16 && nonce.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "worker nonce must be 8-byte hex"
    );
    u64::from_str_radix(nonce, 16).context("parse worker generation from nonce")
}

fn verify_digest(path: &Path, expected: &str) -> Result<()> {
    let actual = sha256_path(path)?;
    ensure!(
        actual == expected,
        "artifact digest mismatch: expected {expected}, got {actual}"
    );
    Ok(())
}

fn sha256_path(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    if path.is_file() {
        hash_file(path, &mut hasher)?;
    } else {
        let mut files = Vec::new();
        collect_files(path, path, &mut files)?;
        files.sort_by(|left, right| left.0.cmp(&right.0));
        for (relative, file) in files {
            hasher.update(relative.as_bytes());
            hasher.update([0]);
            hash_file(&file, &mut hasher)?;
        }
    }
    Ok(hex::encode(hasher.finalize()))
}

fn collect_files(root: &Path, dir: &Path, files: &mut Vec<(String, PathBuf)>) -> Result<()> {
    for entry in
        fs::read_dir(dir).with_context(|| format!("read artifact directory {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, files)?;
        } else if path.is_file() {
            let relative = path
                .strip_prefix(root)
                .context("artifact file escaped root")?
                .to_string_lossy()
                .replace('\\', "/");
            files.push((relative, path));
        }
    }
    Ok(())
}

fn hash_file(path: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut file =
        fs::File::open(path).with_context(|| format!("open artifact {}", path.display()))?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn read_json<T: serde::de::DeserializeOwned>(stream: &mut UnixStream, max_frame: u32) -> Result<T> {
    let bytes = read_frame(stream, max_frame)?;
    serde_json::from_slice(&bytes).context("decode worker JSON frame")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_generation_is_bound_to_nonce() {
        assert_eq!(
            worker_generation("0123456789abcdef").unwrap(),
            0x0123_4567_89ab_cdef
        );
        assert!(worker_generation("not-a-nonce").is_err());
    }

    #[test]
    fn custom_transport_rejects_unknown_fields() {
        let value = serde_json::json!({
            "type": "GENERATE_CANCEL",
            "req_id": "r1",
            "cancellation": { "generation_id": "g1" },
            "raw_schema": {}
        });
        assert!(serde_json::from_value::<DecodeTransportRequest>(value).is_err());
    }

    #[test]
    fn engine_identity_names_fleet_protocol() {
        let identity = engine_identity(7);
        assert_eq!(identity.engine, "owned-metal-decode");
        assert_eq!(
            identity.build_flags["protocol"],
            "owned-metal-decode-worker-v1"
        );
        assert_eq!(identity.build_flags["worker_generation"], "7");
        assert_eq!(
            identity.build_flags["constraint_encoding"],
            "token-id-json-constraint-v1"
        );
    }
}
