use std::{
    collections::BTreeMap,
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
        FrameEnvelope, GenerateContinue, GenerateInstallHintBank, GenerateProgress, GenerateStart,
        HintBankInstalled, TokenIdJsonConstraint, WorkerFrame,
    },
    validation::{validate_start, WorkerStartContext},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use synapse_core::{
    worker_engine_names::DECODE_WORKER_ENGINE,
    worker_framing_sync::{read_frame, write_json_frame},
    worker_protocol::{
        WorkerHello, WorkerHelloAck, WorkerRequest, WorkerResponse, DEFAULT_MAX_FRAME_BYTES,
        WORKER_PROTOCOL_VERSION,
    },
    EngineIdentity, Fingerprint, SidecarHintBank,
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
    owned_decode_sidecar::find_hint_continuation,
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
    #[arg(long, hide = true)]
    test_abort_after_progress: bool,
    #[arg(long, hide = true)]
    test_abort_after_progress_once: Option<PathBuf>,
    /// Test and benchmark escape hatch for exact A/B comparison. Production
    /// leaves the singleton fast path enabled.
    #[arg(long, hide = true)]
    disable_forced_token_fast_path: bool,
}

enum DecodeEngine {
    Qwen {
        decoder: MetalStepDecoder<'static>,
        f16_prefill: Option<MetalStepDecoder<'static>>,
        layer_count: usize,
    },
    Lfm2 {
        engine: Lfm2HybridStepEngine,
    },
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
            Self::Qwen {
                decoder,
                f16_prefill,
                layer_count,
            } => {
                let (cache, token) = if let Some(prefill) = f16_prefill {
                    // Mirror bench/spikes/unified-rt/src/qwen3_decode_metal_step.rs:
                    // Q8 keeps prefill at f16 quality, then hands the exact f16 KV
                    // bits to the bandwidth-oriented Q8 step engine.
                    let (prefill_cache, token) = prefill.prefill(prompt)?;
                    let cache =
                        handoff_qwen_cache(prefill, decoder, *layer_count, prefill_cache.position)?;
                    (cache, token)
                } else {
                    decoder.prefill(prompt)?
                };
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
            Self::Qwen {
                decoder,
                f16_prefill,
                layer_count,
            } => {
                let mut cache = MetalStepKvCache { position: 0 };
                let mut logits = Vec::new();
                if let Some(prefill) = f16_prefill {
                    for &token in prompt {
                        logits = prefill.advance(&mut cache, token)?;
                    }
                    cache = handoff_qwen_cache(prefill, decoder, *layer_count, cache.position)?;
                } else {
                    for &token in prompt {
                        logits = decoder.advance(&mut cache, token)?;
                    }
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
            (Self::Qwen { decoder, .. }, DecodeCache::Qwen(cache)) => decoder.advance(cache, token),
            (Self::Lfm2 { engine }, DecodeCache::Lfm2(cache)) => engine.advance(cache, token),
            _ => bail!("owned decode engine/cache family mismatch"),
        }
    }

    /// Ingest committed tokens and return logits after the final token. Qwen
    /// reuses the 16-position batched verifier; LFM2 uses its verifier for the
    /// prefix and materializes logits only for the final position.
    fn ingest_tokens_for_logits(
        &mut self,
        cache: &mut DecodeCache,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "forced-token ingest requires tokens");
        match (self, cache) {
            (Self::Qwen { decoder, .. }, DecodeCache::Qwen(cache)) => {
                let mut chunks = tokens.chunks(PRODUCTION_N as usize).peekable();
                while let Some(chunk) = chunks.next() {
                    if chunks.peek().is_some() {
                        decoder.verify_tokens_batch(cache, chunk)?;
                        continue;
                    }
                    let mut logits = decoder.verify_tokens_batch_logits(cache, chunk)?;
                    let row_len = logits.len() / chunk.len();
                    return Ok(logits.split_off(logits.len() - row_len));
                }
                unreachable!("non-empty token span has a final chunk")
            }
            (Self::Lfm2 { engine }, DecodeCache::Lfm2(cache)) => {
                let (&last, prefix) = tokens
                    .split_last()
                    .expect("non-empty token span has a final token");
                if !prefix.is_empty() {
                    engine.verify_tokens(cache, prefix)?;
                }
                engine.advance(cache, last)
            }
            _ => bail!("owned decode engine/cache family mismatch"),
        }
    }

    /// Verify a bounded proposal and return one full logits row after each
    /// supplied token. Qwen executes the production batched primitive; LFM2
    /// retains the same row alignment through its resident sequential step path.
    fn verify_tokens_batch_logits(
        &mut self,
        cache: &mut DecodeCache,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= PRODUCTION_N as usize,
            "sidecar verification requires between one and {PRODUCTION_N} tokens"
        );
        match (self, cache) {
            (Self::Qwen { decoder, .. }, DecodeCache::Qwen(cache)) => {
                decoder.verify_tokens_batch_logits(cache, tokens)
            }
            (Self::Lfm2 { engine }, DecodeCache::Lfm2(cache)) => {
                let mut rows = Vec::new();
                for &token in tokens {
                    rows.extend(engine.advance(cache, token)?);
                }
                Ok(rows)
            }
            _ => bail!("owned decode engine/cache family mismatch"),
        }
    }

    fn rewind(&mut self, cache: &mut DecodeCache, position: usize) -> Result<()> {
        match (self, cache) {
            (Self::Qwen { decoder, .. }, DecodeCache::Qwen(cache)) => {
                decoder.rewind(cache, position)
            }
            (Self::Lfm2 { engine }, DecodeCache::Lfm2(cache)) => engine.rewind(cache, position),
            _ => bail!("owned decode engine/cache family mismatch"),
        }
    }

    fn cache_position(&self, cache: &DecodeCache) -> Result<usize> {
        match (self, cache) {
            (Self::Qwen { .. }, DecodeCache::Qwen(cache)) => Ok(cache.position),
            (Self::Lfm2 { .. }, DecodeCache::Lfm2(cache)) => Ok(cache.position),
            _ => bail!("owned decode engine/cache family mismatch"),
        }
    }

    fn chain_span(&self) -> usize {
        match self {
            Self::Qwen { decoder, .. } => decoder.chain_span(),
            Self::Lfm2 { engine } => engine.chain_span(),
        }
    }

    fn set_chain_span(&mut self, span: usize) -> Result<()> {
        match self {
            Self::Qwen {
                decoder,
                f16_prefill,
                ..
            } => {
                decoder.set_chain_span(span)?;
                if let Some(prefill) = f16_prefill {
                    prefill.set_chain_span(span)?;
                }
                Ok(())
            }
            Self::Lfm2 { engine } => engine.set_chain_span(span),
        }
    }

    fn advance_chain(
        &mut self,
        cache: &mut DecodeCache,
        seed: u32,
        steps: usize,
    ) -> Result<Vec<u32>> {
        match (self, cache) {
            (Self::Qwen { decoder, .. }, DecodeCache::Qwen(cache)) => {
                decoder.advance_chain(cache, seed, steps)
            }
            (Self::Lfm2 { engine }, DecodeCache::Lfm2(cache)) => {
                engine.advance_chain(cache, seed, steps)
            }
            _ => bail!("owned decode engine/cache family mismatch"),
        }
    }
}

fn handoff_qwen_cache(
    prefill: &MetalStepDecoder<'_>,
    decoder: &mut MetalStepDecoder<'_>,
    layer_count: usize,
    position: usize,
) -> Result<MetalStepKvCache> {
    let mut cache_bits = Vec::new();
    for layer in 0..layer_count {
        cache_bits.extend(prefill.inspect_cache_bits(layer)?);
    }
    decoder.import_caches(&cache_bits)?;
    Ok(MetalStepKvCache { position })
}

struct LoadedRuntime {
    model_ref: String,
    decode_fingerprint: String,
    runtime_config_digest: String,
    production_n: u32,
    /// Machine-configured free-text chain span. Constraint requests override it
    /// to one at start because host-side token masking needs per-token logits.
    decode_chain_k: usize,
    stop_ids: Vec<u32>,
    vocabulary: Arc<TokenVocabulary>,
    vocabulary_digest: String,
    context_bucket: usize,
    engine: DecodeEngine,
}

struct ActiveConstraint {
    automaton: Automaton,
    state: State,
    identity: String,
    schema_identity: String,
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
    /// Committed constrained tokens whose KV updates are intentionally deferred
    /// until a non-singleton mask needs real logits.
    pending_ingest_ids: Vec<u32>,
    /// Complete target-tokenized sidecar bank, installed only at a progress
    /// boundary through the owned-decode transport.
    sidecar_hint_bank: Option<SidecarHintBank>,
    /// The worker accepts bank installation only while it is paused after a
    /// progress frame and before the matching continue request.
    awaiting_continue: bool,
}

struct HintVerification {
    committed: usize,
    terminal: Option<FinishReason>,
}

struct WorkerState {
    worker_generation: u64,
    forced_token_fast_path: bool,
    loaded: Option<LoadedRuntime>,
    resident: Option<ResidentGeneration>,
}

impl WorkerState {
    fn new(worker_generation: u64, forced_token_fast_path: bool) -> Self {
        Self {
            worker_generation,
            forced_token_fast_path,
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
        let decode_chain_k = runtime_config
            .get("decode_chain_k")
            .map(String::as_str)
            .unwrap_or("1")
            .parse::<usize>()
            .context("parse owned decode decode_chain_k")?;
        ensure!(
            (1..=16).contains(&decode_chain_k),
            "decode_chain_k must be between 1 and 16"
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
                let layer_count = model.layers.len();
                let decoder = MetalStepDecoder::new(model, Precision::F16, bucket, quant)?;
                let f16_prefill = if quant.is_quantized() {
                    let model = Box::leak(Box::new(Qwen3DecodeModel::load_with_quant(
                        path,
                        Precision::F16,
                        WeightQuantization::None,
                    )?));
                    Some(MetalStepDecoder::new(
                        model,
                        Precision::F16,
                        bucket,
                        WeightQuantization::None,
                    )?)
                } else {
                    None
                };
                (
                    DecodeEngine::Qwen {
                        decoder,
                        f16_prefill,
                        layer_count,
                    },
                    stop_ids,
                )
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
            decode_chain_k,
            stop_ids,
            vocabulary,
            vocabulary_digest,
            context_bucket: bucket,
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
        let effective_chain_k =
            effective_decode_chain_k(loaded.decode_chain_k, active_constraint.is_some());
        loaded
            .engine
            .set_chain_span(effective_chain_k)
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
            pending_ingest_ids: Vec::new(),
            sidecar_hint_bank: None,
            awaiting_continue: false,
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
        self.resident
            .as_mut()
            .ok_or(DecodeError::ProtocolMismatch)?
            .awaiting_continue = false;
        self.run_quantum(continuation.next_token_budget)
    }

    fn install_hint_bank(
        &mut self,
        installation: GenerateInstallHintBank,
    ) -> Result<HintBankInstalled, DecodeError> {
        let resident = self
            .resident
            .as_mut()
            .ok_or(DecodeError::ProtocolMismatch)?;
        let loaded = self
            .loaded
            .as_ref()
            .ok_or(DecodeError::RuntimeConfigMismatch)?;
        if !resident.awaiting_continue || installation.generation_id != resident.generation_id {
            return Err(DecodeError::ProtocolMismatch);
        }
        let constraint = resident
            .constraint
            .as_ref()
            .ok_or(DecodeError::ProtocolMismatch)?;
        if installation.bank.schema_identity != constraint_schema_identity(constraint)
            || !hint_bank_is_bounded(&installation.bank, loaded.vocabulary.len())
        {
            return Err(DecodeError::ProtocolMismatch);
        }
        let digest = install_sidecar_bank(&mut resident.sidecar_hint_bank, installation.bank)?;
        Ok(HintBankInstalled {
            generation_id: resident.generation_id.clone(),
            bank_content_digest: digest,
        })
    }

    fn run_quantum(&mut self, token_budget: u32) -> Result<WorkerFrame, DecodeError> {
        if self
            .resident
            .as_ref()
            .is_some_and(|resident| resident.constraint.is_none())
            && self
                .loaded
                .as_ref()
                .is_some_and(|loaded| loaded.engine.chain_span() > 1)
        {
            return self.run_chained_quantum(token_budget);
        }
        self.run_single_quantum(token_budget)
    }

    fn run_single_quantum(&mut self, token_budget: u32) -> Result<WorkerFrame, DecodeError> {
        let mut remaining_quantum = token_budget as usize;
        while remaining_quantum > 0 {
            // Grammar positions with one legal token advance without model logits
            // before the worker considers any sidecar continuation.
            let forced = {
                let loaded = self
                    .loaded
                    .as_ref()
                    .ok_or(DecodeError::RuntimeConfigMismatch)?;
                let resident = self
                    .resident
                    .as_mut()
                    .ok_or(DecodeError::ProtocolMismatch)?;
                (self.forced_token_fast_path && resident.constraint.is_some())
                    .then(|| {
                        resident.constraint.as_ref().and_then(|constraint| {
                            sole_survivor(&loaded.stop_ids, &*loaded.vocabulary, constraint)
                        })
                    })
                    .flatten()
            };

            if forced.is_none() {
                self.flush_pending_constrained_ingest()?;
                if let Some(verification) =
                    self.try_verify_sidecar_hint(remaining_quantum.min(PRODUCTION_N as usize))?
                {
                    if verification.committed == 0 || verification.committed > remaining_quantum {
                        return Err(DecodeError::ProtocolMismatch);
                    }
                    remaining_quantum -= verification.committed;
                    if let Some(reason) = verification.terminal {
                        return Ok(self.finish(reason));
                    }
                    continue;
                }
            }

            let defer_ingest = self.resident.as_ref().is_some_and(|resident| {
                self.forced_token_fast_path && resident.constraint.is_some()
            });
            let token = match forced {
                Some(token) => {
                    // Held logits describe the prefix before the forced run and
                    // cannot select the next target-evaluated position after the
                    // parser commits this token.
                    self.resident
                        .as_mut()
                        .ok_or(DecodeError::ProtocolMismatch)?
                        .next_logits = None;
                    token
                }
                None => {
                    let loaded = self
                        .loaded
                        .as_ref()
                        .ok_or(DecodeError::RuntimeConfigMismatch)?;
                    let resident = self
                        .resident
                        .as_mut()
                        .ok_or(DecodeError::ProtocolMismatch)?;
                    if let Some(constraint) = resident.constraint.as_ref() {
                        let logits = resident
                            .next_logits
                            .take()
                            .ok_or(DecodeError::ProtocolMismatch)?;
                        constrained_top1(
                            &logits,
                            &loaded.stop_ids,
                            &*loaded.vocabulary,
                            constraint,
                        )?
                    } else {
                        resident
                            .next_greedy
                            .take()
                            .ok_or(DecodeError::ProtocolMismatch)?
                    }
                }
            };

            let terminal = {
                let loaded = self
                    .loaded
                    .as_ref()
                    .ok_or(DecodeError::RuntimeConfigMismatch)?;
                let resident = self
                    .resident
                    .as_mut()
                    .ok_or(DecodeError::ProtocolMismatch)?;
                if loaded.stop_ids.contains(&token) {
                    if let Some(constraint) = resident.constraint.as_ref() {
                        if constraint.automaton.has_complete_value(&constraint.state) {
                            Some(FinishReason::GrammarComplete)
                        } else {
                            self.resident = None;
                            return Err(DecodeError::GrammarStopBeforeCompletion);
                        }
                    } else {
                        Some(FinishReason::StopToken)
                    }
                } else {
                    if let Some(constraint) = resident.constraint.as_mut() {
                        commit_constrained_token(constraint, &*loaded.vocabulary, token)?;
                    }
                    resident.generated_ids.push(token);
                    if resident.constraint.as_ref().is_some_and(|constraint| {
                        constraint.automaton.has_complete_value(&constraint.state)
                    }) {
                        Some(FinishReason::GrammarComplete)
                    } else if resident.generated_ids.len() as u32 == resident.max_tokens {
                        if resident.constraint.is_some() {
                            self.resident = None;
                            return Err(DecodeError::GrammarMaxTokensExhausted);
                        }
                        Some(FinishReason::MaxTokens)
                    } else {
                        None
                    }
                }
            };
            remaining_quantum -= 1;
            if let Some(reason) = terminal {
                return Ok(self.finish(reason));
            }

            if defer_ingest {
                self.resident
                    .as_mut()
                    .ok_or(DecodeError::ProtocolMismatch)?
                    .pending_ingest_ids
                    .push(token);
                continue;
            }
            let logits = {
                let loaded = self
                    .loaded
                    .as_mut()
                    .ok_or(DecodeError::RuntimeConfigMismatch)?;
                let resident = self
                    .resident
                    .as_mut()
                    .ok_or(DecodeError::ProtocolMismatch)?;
                loaded
                    .engine
                    .advance(&mut resident.cache, token)
                    .map_err(|_| DecodeError::Unavailable)?
            };
            let resident = self
                .resident
                .as_mut()
                .ok_or(DecodeError::ProtocolMismatch)?;
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
        resident.awaiting_continue = true;
        Ok(WorkerFrame::Progress(GenerateProgress {
            generation_id: resident.generation_id.clone(),
            quantum_sequence: resident.quantum_sequence,
            committed_token_count: resident.generated_ids.len() as u32,
        }))
    }

    fn flush_pending_constrained_ingest(&mut self) -> Result<(), DecodeError> {
        let pending = self
            .resident
            .as_mut()
            .ok_or(DecodeError::ProtocolMismatch)?
            .pending_ingest_ids
            .drain(..)
            .collect::<Vec<_>>();
        if pending.is_empty() {
            return Ok(());
        }
        let logits = {
            let loaded = self
                .loaded
                .as_mut()
                .ok_or(DecodeError::RuntimeConfigMismatch)?;
            let resident = self
                .resident
                .as_mut()
                .ok_or(DecodeError::ProtocolMismatch)?;
            loaded
                .engine
                .ingest_tokens_for_logits(&mut resident.cache, &pending)
                .map_err(|_| DecodeError::Unavailable)?
        };
        self.resident
            .as_mut()
            .ok_or(DecodeError::ProtocolMismatch)?
            .next_logits = Some(logits);
        Ok(())
    }

    /// Verify a selected bank continuation with the grammar state at every
    /// proposed position. Position zero uses resident next logits; later positions
    /// use verifier row `i - 1`. A mismatch rewinds the speculative cache and
    /// replays the accepted prefix plus the target-selected legal token.
    fn try_verify_sidecar_hint(
        &mut self,
        proposal_limit: usize,
    ) -> Result<Option<HintVerification>, DecodeError> {
        let continuation = {
            let loaded = self
                .loaded
                .as_ref()
                .ok_or(DecodeError::RuntimeConfigMismatch)?;
            let resident = self
                .resident
                .as_ref()
                .ok_or(DecodeError::ProtocolMismatch)?;
            let Some(bank) = resident.sidecar_hint_bank.as_ref() else {
                return Ok(None);
            };
            if resident.constraint.is_none() {
                return Ok(None);
            }
            let cache_position = loaded
                .engine
                .cache_position(&resident.cache)
                .map_err(|_| DecodeError::Unavailable)?;
            find_hint_continuation(
                bank,
                &resident.generated_ids,
                loaded.context_bucket.saturating_sub(cache_position),
                resident
                    .max_tokens
                    .saturating_sub(resident.generated_ids.len() as u32) as usize,
                proposal_limit,
            )
        };
        let Some(continuation) = continuation else {
            return Ok(None);
        };

        let (checkpoint_position, resident_logits, initial_state, vocabulary_len) = {
            let loaded = self
                .loaded
                .as_ref()
                .ok_or(DecodeError::RuntimeConfigMismatch)?;
            let resident = self
                .resident
                .as_ref()
                .ok_or(DecodeError::ProtocolMismatch)?;
            let constraint = resident
                .constraint
                .as_ref()
                .ok_or(DecodeError::ProtocolMismatch)?;
            (
                loaded
                    .engine
                    .cache_position(&resident.cache)
                    .map_err(|_| DecodeError::Unavailable)?,
                resident
                    .next_logits
                    .clone()
                    .ok_or(DecodeError::ProtocolMismatch)?,
                constraint.state.clone(),
                loaded.vocabulary.len(),
            )
        };
        let verifier_logits = {
            let loaded = self
                .loaded
                .as_mut()
                .ok_or(DecodeError::RuntimeConfigMismatch)?;
            let resident = self
                .resident
                .as_mut()
                .ok_or(DecodeError::ProtocolMismatch)?;
            loaded
                .engine
                .verify_tokens_batch_logits(&mut resident.cache, &continuation.tokens)
                .map_err(|_| DecodeError::Unavailable)?
        };
        if verifier_logits.len() != continuation.tokens.len() * vocabulary_len {
            return Err(DecodeError::ProtocolMismatch);
        }

        let (accepted, mismatch, terminal, accepted_state) = {
            let loaded = self
                .loaded
                .as_ref()
                .ok_or(DecodeError::RuntimeConfigMismatch)?;
            let resident = self
                .resident
                .as_ref()
                .ok_or(DecodeError::ProtocolMismatch)?;
            let constraint = resident
                .constraint
                .as_ref()
                .ok_or(DecodeError::ProtocolMismatch)?;
            let mut state = initial_state;
            let mut accepted = 0usize;
            let mut mismatch = None;
            let mut terminal = None;
            for (index, &proposed) in continuation.tokens.iter().enumerate() {
                let row = if index == 0 {
                    &resident_logits
                } else {
                    &verifier_logits[(index - 1) * vocabulary_len..index * vocabulary_len]
                };
                let expected = constrained_top1_at_state(
                    row,
                    &loaded.stop_ids,
                    &*loaded.vocabulary,
                    &constraint.automaton,
                    &state,
                )?;
                if expected != proposed {
                    mismatch = Some(expected);
                    break;
                }
                if loaded.stop_ids.contains(&expected) {
                    terminal = Some(if constraint.automaton.has_complete_value(&state) {
                        FinishReason::GrammarComplete
                    } else {
                        return Err(DecodeError::GrammarStopBeforeCompletion);
                    });
                    break;
                }
                state = commit_constraint_state(
                    &constraint.automaton,
                    &*loaded.vocabulary,
                    state,
                    expected,
                )?;
                accepted += 1;
                if constraint.automaton.has_complete_value(&state) {
                    terminal = Some(FinishReason::GrammarComplete);
                    break;
                }
            }
            (accepted, mismatch, terminal, state)
        };

        let full_acceptance =
            mismatch.is_none() && terminal.is_none() && accepted == continuation.tokens.len();
        if full_acceptance {
            let final_logits =
                verifier_logits[(continuation.tokens.len() - 1) * vocabulary_len..].to_vec();
            let resident = self
                .resident
                .as_mut()
                .ok_or(DecodeError::ProtocolMismatch)?;
            resident
                .generated_ids
                .extend_from_slice(&continuation.tokens);
            resident
                .constraint
                .as_mut()
                .ok_or(DecodeError::ProtocolMismatch)?
                .state = accepted_state;
            resident.next_logits = Some(final_logits);
            let exhausted = resident.generated_ids.len() as u32 == resident.max_tokens;
            if exhausted {
                // A constrained generation that reaches its configured budget
                // before completing is a typed terminal failure, not a resident
                // generation waiting for another continuation.
                self.resident = None;
                return Err(DecodeError::GrammarMaxTokensExhausted);
            }
            return Ok(Some(HintVerification {
                committed: continuation.tokens.len(),
                terminal: None,
            }));
        }

        // The batched primitive advanced through the entire proposal. Restore the
        // pre-verification checkpoint and replay the same state reached by an
        // ordinary constrained greedy decode.
        let mismatch_is_stop = mismatch.is_some_and(|token| {
            self.loaded
                .as_ref()
                .is_some_and(|loaded| loaded.stop_ids.contains(&token))
        });
        let replay =
            verification_replay_tokens(&continuation.tokens, accepted, mismatch, mismatch_is_stop);
        let mut next_logits = None;
        {
            let loaded = self
                .loaded
                .as_mut()
                .ok_or(DecodeError::RuntimeConfigMismatch)?;
            let resident = self
                .resident
                .as_mut()
                .ok_or(DecodeError::ProtocolMismatch)?;
            loaded
                .engine
                .rewind(&mut resident.cache, checkpoint_position)
                .map_err(|_| DecodeError::Unavailable)?;
            for token in &replay {
                next_logits = Some(
                    loaded
                        .engine
                        .advance(&mut resident.cache, *token)
                        .map_err(|_| DecodeError::Unavailable)?,
                );
            }
        }
        let resident = self
            .resident
            .as_mut()
            .ok_or(DecodeError::ProtocolMismatch)?;
        resident
            .generated_ids
            .extend_from_slice(&continuation.tokens[..accepted]);
        resident
            .constraint
            .as_mut()
            .ok_or(DecodeError::ProtocolMismatch)?
            .state = accepted_state;
        if let Some(token) = mismatch {
            let loaded = self
                .loaded
                .as_ref()
                .ok_or(DecodeError::RuntimeConfigMismatch)?;
            if mismatch_is_stop {
                let reason = if resident.constraint.as_ref().is_some_and(|constraint| {
                    constraint.automaton.has_complete_value(&constraint.state)
                }) {
                    FinishReason::GrammarComplete
                } else {
                    return Err(DecodeError::GrammarStopBeforeCompletion);
                };
                return Ok(Some(HintVerification {
                    committed: accepted,
                    terminal: Some(reason),
                }));
            }
            let constraint = resident
                .constraint
                .as_mut()
                .ok_or(DecodeError::ProtocolMismatch)?;
            commit_constrained_token(constraint, &*loaded.vocabulary, token)?;
            resident.generated_ids.push(token);
        }
        resident.next_logits = next_logits;
        let committed = replay.len();
        let terminal = terminal.or_else(|| {
            resident
                .constraint
                .as_ref()
                .is_some_and(|constraint| {
                    constraint.automaton.has_complete_value(&constraint.state)
                })
                .then_some(FinishReason::GrammarComplete)
        });
        let exhausted =
            terminal.is_none() && resident.generated_ids.len() as u32 == resident.max_tokens;
        if exhausted {
            self.resident = None;
            return Err(DecodeError::GrammarMaxTokensExhausted);
        }
        Ok(Some(HintVerification {
            committed,
            terminal,
        }))
    }

    fn commit_unconstrained_token(
        &mut self,
        token: u32,
    ) -> Result<Option<WorkerFrame>, DecodeError> {
        let stop_ids = self
            .loaded
            .as_ref()
            .ok_or(DecodeError::RuntimeConfigMismatch)?
            .stop_ids
            .clone();
        let resident = self
            .resident
            .as_mut()
            .ok_or(DecodeError::ProtocolMismatch)?;
        if stop_ids.contains(&token) {
            return Ok(Some(self.finish(FinishReason::StopToken)));
        }
        resident.generated_ids.push(token);
        if resident.generated_ids.len() as u32 == resident.max_tokens {
            return Ok(Some(self.finish(FinishReason::MaxTokens)));
        }
        Ok(None)
    }

    /// Run an unconstrained quantum with fused chain submissions while keeping
    /// the scheduler-visible budget at exactly `token_budget`. The final chain
    /// output is retained as the next prediction, just like the single-step
    /// path's post-token logits, so no extra token is committed at a boundary.
    fn run_chained_quantum(&mut self, token_budget: u32) -> Result<WorkerFrame, DecodeError> {
        let chain_k = self
            .loaded
            .as_ref()
            .ok_or(DecodeError::RuntimeConfigMismatch)?
            .engine
            .chain_span();
        let mut seed = self
            .resident
            .as_mut()
            .ok_or(DecodeError::ProtocolMismatch)?
            .next_greedy
            .take()
            .ok_or(DecodeError::ProtocolMismatch)?;
        if let Some(frame) = self.commit_unconstrained_token(seed)? {
            return Ok(frame);
        }
        let mut committed = 1_u32;
        let mut forward_steps = 0_u32;
        while forward_steps < token_budget {
            let steps = chain_k.min((token_budget - forward_steps) as usize);
            let tokens = {
                let loaded = self
                    .loaded
                    .as_mut()
                    .ok_or(DecodeError::RuntimeConfigMismatch)?;
                let resident = self
                    .resident
                    .as_mut()
                    .ok_or(DecodeError::ProtocolMismatch)?;
                loaded
                    .engine
                    .advance_chain(&mut resident.cache, seed, steps)
                    .map_err(|_| DecodeError::Unavailable)?
            };
            forward_steps += steps as u32;
            for token in tokens {
                if committed < token_budget {
                    if let Some(frame) = self.commit_unconstrained_token(token)? {
                        return Ok(frame);
                    }
                    committed += 1;
                    seed = token;
                } else {
                    let resident = self
                        .resident
                        .as_mut()
                        .ok_or(DecodeError::ProtocolMismatch)?;
                    resident.next_greedy = Some(token);
                    resident.quantum_sequence = resident.quantum_sequence.saturating_add(1);
                    resident.awaiting_continue = true;
                    return Ok(WorkerFrame::Progress(GenerateProgress {
                        generation_id: resident.generation_id.clone(),
                        quantum_sequence: resident.quantum_sequence,
                        committed_token_count: resident.generated_ids.len() as u32,
                    }));
                }
            }
        }
        Err(DecodeError::ProtocolMismatch)
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
            constraint_complete: resident.constraint.as_ref().is_some_and(|constraint| {
                constraint.automaton.has_complete_value(&constraint.state)
            }),
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
    let mut state = WorkerState::new(worker_generation, !args.disable_forced_token_fast_path);

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
            handle_decode_request(
                &mut stream,
                max_frame,
                &mut state,
                request,
                args.test_abort_after_progress,
                args.test_abort_after_progress_once.as_deref(),
            )?;
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
    abort_after_progress: bool,
    abort_after_progress_once: Option<&Path>,
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
        DecodeTransportRequest::GenerateInstallHintBank {
            req_id,
            installation,
        } => match state.install_hint_bank(installation) {
            Ok(installation) => DecodeTransportResponse::HintBankInstalled {
                req_id,
                installation,
            },
            Err(error) => DecodeTransportResponse::Frame {
                req_id,
                envelope: error_frame(error),
            },
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
    let emitted_progress = matches!(
        &response,
        DecodeTransportResponse::Frame {
            envelope: FrameEnvelope {
                frame: WorkerFrame::Progress(_),
                ..
            },
            ..
        }
    );
    let abort_once = emitted_progress
        && abort_after_progress_once.is_some_and(|marker| {
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(marker)
                .is_ok()
        });
    if emitted_progress && (abort_after_progress || abort_once) {
        std::process::abort();
    }
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
        schema_identity: constraint.canonical_schema_digest.clone(),
    })
}

trait ConstraintVocabulary {
    fn len(&self) -> usize;
    fn token_piece(&self, token_id: u32) -> Option<&[u8]>;
}

impl ConstraintVocabulary for TokenVocabulary {
    fn len(&self) -> usize {
        TokenVocabulary::len(self)
    }

    fn token_piece(&self, token_id: u32) -> Option<&[u8]> {
        TokenVocabulary::token_piece(self, token_id)
    }
}

fn constrained_token_is_permitted<V: ConstraintVocabulary + ?Sized>(
    token_id: u32,
    stop_ids: &[u32],
    vocabulary: &V,
    constraint: &ActiveConstraint,
) -> bool {
    constrained_token_is_permitted_at_state(
        token_id,
        stop_ids,
        vocabulary,
        &constraint.automaton,
        &constraint.state,
    )
}

fn constrained_token_is_permitted_at_state<V: ConstraintVocabulary + ?Sized>(
    token_id: u32,
    stop_ids: &[u32],
    vocabulary: &V,
    automaton: &Automaton,
    state: &State,
) -> bool {
    if stop_ids.contains(&token_id) {
        return automaton.has_complete_value(state);
    }
    vocabulary
        .token_piece(token_id)
        .is_some_and(|piece| automaton.token_is_decode_permitted(state, piece))
}

fn commit_constraint_state<V: ConstraintVocabulary + ?Sized>(
    automaton: &Automaton,
    vocabulary: &V,
    state: State,
    token: u32,
) -> Result<State, DecodeError> {
    let piece = vocabulary
        .token_piece(token)
        .ok_or(DecodeError::GrammarUnsatisfiable)?;
    automaton
        .commit_token(&state, piece)
        .map_err(|_| DecodeError::GrammarUnsatisfiable)
}

fn commit_constrained_token<V: ConstraintVocabulary + ?Sized>(
    constraint: &mut ActiveConstraint,
    vocabulary: &V,
    token: u32,
) -> Result<(), DecodeError> {
    constraint.state = commit_constraint_state(
        &constraint.automaton,
        vocabulary,
        constraint.state.clone(),
        token,
    )?;
    Ok(())
}

fn sole_survivor<V: ConstraintVocabulary + ?Sized>(
    stop_ids: &[u32],
    vocabulary: &V,
    constraint: &ActiveConstraint,
) -> Option<u32> {
    let mut survivor = None;
    for token_id in 0..vocabulary.len() as u32 {
        if !constrained_token_is_permitted(token_id, stop_ids, vocabulary, constraint) {
            continue;
        }
        if survivor.replace(token_id).is_some() {
            return None;
        }
    }
    survivor
}

fn constraint_schema_identity(constraint: &ActiveConstraint) -> &str {
    &constraint.schema_identity
}

fn hint_bank_is_bounded(bank: &SidecarHintBank, vocabulary_len: usize) -> bool {
    const MAX_VIEWS: usize = 2;
    const MAX_TOKENS_PER_VIEW: usize = 4_096;

    !bank.views.is_empty()
        && bank.views.len() <= MAX_VIEWS
        && bank.views.iter().all(|view| {
            !view.is_empty()
                && view.len() <= MAX_TOKENS_PER_VIEW
                && view.iter().all(|&token| (token as usize) < vocabulary_len)
        })
}

fn verification_replay_tokens(
    proposal: &[u32],
    accepted: usize,
    mismatch: Option<u32>,
    mismatch_is_stop: bool,
) -> Vec<u32> {
    let mut replay = proposal[..accepted].to_vec();
    if let Some(token) = mismatch.filter(|_| !mismatch_is_stop) {
        replay.push(token);
    }
    replay
}

fn install_sidecar_bank(
    installed: &mut Option<SidecarHintBank>,
    bank: SidecarHintBank,
) -> Result<String, DecodeError> {
    let digest = bank.content_digest();
    match installed.as_ref() {
        Some(existing) if existing.content_digest() == digest => Ok(digest),
        Some(_) => Err(DecodeError::ProtocolMismatch),
        None => {
            *installed = Some(bank);
            Ok(digest)
        }
    }
}

fn constrained_top1<V: ConstraintVocabulary + ?Sized>(
    logits: &[f32],
    stop_ids: &[u32],
    vocabulary: &V,
    constraint: &ActiveConstraint,
) -> Result<u32, DecodeError> {
    constrained_top1_at_state(
        logits,
        stop_ids,
        vocabulary,
        &constraint.automaton,
        &constraint.state,
    )
}

fn constrained_top1_at_state<V: ConstraintVocabulary + ?Sized>(
    logits: &[f32],
    stop_ids: &[u32],
    vocabulary: &V,
    automaton: &Automaton,
    state: &State,
) -> Result<u32, DecodeError> {
    let mut selected: Option<(u32, f32)> = None;
    for (index, &logit) in logits.iter().enumerate() {
        let token_id = index as u32;
        if !constrained_token_is_permitted_at_state(
            token_id, stop_ids, vocabulary, automaton, state,
        ) {
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

/// Build the identity announced in the worker HELLO handshake.
pub fn engine_identity(worker_generation: u64) -> EngineIdentity {
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
        engine: DECODE_WORKER_ENGINE.to_string(),
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

fn effective_decode_chain_k(configured: usize, constrained: bool) -> usize {
    if constrained {
        1
    } else {
        configured
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
    fn producer_identity_matches_shared_catalog_constant() {
        assert_eq!(engine_identity(7).engine, DECODE_WORKER_ENGINE);
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
    fn producer_identity_matches_catalog_engine_and_fleet_protocol() {
        let identity = engine_identity(7);
        assert_eq!(identity.engine, DECODE_WORKER_ENGINE);
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

#[cfg(test)]
mod fast_path_tests {
    use std::{
        collections::BTreeMap,
        path::{Path, PathBuf},
        time::Instant,
    };

    use owned_decode_worker::protocol::{
        GenerateContinue, GenerateInstallHintBank, GenerateStart, Sampling, TokenIdJsonConstraint,
        WorkerFrame,
    };
    use synapse_core::{Fingerprint, SidecarHintBank};
    use synapse_module::owned_decode_grammar_scheduler::{
        compile_grammar, grammar_automaton::Automaton, grammar_limits::GrammarLimits,
        grammar_schema::parse_schema, CompileContext, GrammarSubsetManifest,
        TokenIdJsonConstraintV1,
    };

    use super::{
        commit_constraint_state, constrained_top1, constrained_top1_at_state,
        effective_decode_chain_k, hint_bank_is_bounded, install_sidecar_bank, sha256_path,
        sole_survivor, verification_replay_tokens, ActiveConstraint, ConstraintVocabulary,
        DecodeError, TokenVocabulary, Tokenizer, WorkerState, PRODUCTION_N,
    };

    const STOP_ID: u32 = 128;

    struct ByteVocabulary {
        pieces: Vec<Option<Vec<u8>>>,
    }

    impl ByteVocabulary {
        fn ascii() -> Self {
            let mut pieces = (0_u8..=127)
                .map(|byte| Some(vec![byte]))
                .collect::<Vec<_>>();
            pieces.push(None);
            Self { pieces }
        }
    }

    impl ConstraintVocabulary for ByteVocabulary {
        fn len(&self) -> usize {
            self.pieces.len()
        }

        fn token_piece(&self, token_id: u32) -> Option<&[u8]> {
            self.pieces
                .get(token_id as usize)
                .and_then(Option::as_deref)
        }
    }

    struct SimulatedDecode {
        token_ids: Vec<u32>,
        forced_tokens: usize,
        progress_counts: Vec<u32>,
    }

    fn logits(vocabulary_len: usize) -> Vec<f32> {
        (0..vocabulary_len)
            .map(|token_id| match token_id as u8 {
                b'"' => 1_000.0,
                b'}' => 900.0,
                b']' => 800.0,
                b'n' => 700.0,
                b'f' => 600.0,
                b't' => 500.0,
                byte => -(byte as f32),
            })
            .collect()
    }

    fn simulate(schema: &str, fast_path: bool, quantum: usize) -> SimulatedDecode {
        let parsed = parse_schema(schema, &GrammarLimits::default()).expect("schema parses");
        let automaton = Automaton::new(parsed);
        let vocabulary = ByteVocabulary::ascii();
        let mut constraint = ActiveConstraint {
            state: automaton.initial(),
            automaton,
            identity: "test-constraint".to_string(),
            schema_identity: "test-schema".to_string(),
        };
        let scores = logits(vocabulary.len());
        let mut token_ids = Vec::new();
        let mut forced_tokens = 0;
        let mut progress_counts = Vec::new();

        for _ in 0..1_024 {
            let forced = fast_path
                .then(|| sole_survivor(&[STOP_ID], &vocabulary, &constraint))
                .flatten();
            let token = if let Some(token) = forced {
                forced_tokens += 1;
                token
            } else {
                constrained_top1(&scores, &[STOP_ID], &vocabulary, &constraint)
                    .expect("schema has a permitted continuation")
            };
            assert_ne!(token, STOP_ID, "completed values finish before EOS commit");
            let piece = vocabulary
                .token_piece(token)
                .expect("content token has bytes");
            constraint.state = constraint
                .automaton
                .commit_token(&constraint.state, piece)
                .expect("selected token commits");
            token_ids.push(token);
            if constraint.automaton.has_complete_value(&constraint.state) {
                return SimulatedDecode {
                    token_ids,
                    forced_tokens,
                    progress_counts,
                };
            }
            if token_ids.len() % quantum == 0 {
                progress_counts.push(token_ids.len() as u32);
            }
        }
        panic!("simulated constrained decode did not complete");
    }

    fn adversarial_schema_battery() -> Vec<String> {
        (0..30)
            .map(|index| match index % 5 {
                0 => format!(
                    r#"{{"type":"object","properties":{{"field_{index}":{{"type":"string","enum":["value_{index}"]}}}},"required":["field_{index}"],"additionalProperties":false}}"#
                ),
                1 => format!(
                    r#"{{"type":"object","properties":{{"outer_{index}":{{"type":"object","properties":{{"inner_{index}":{{"type":"null"}}}},"required":["inner_{index}"],"additionalProperties":false}}}},"required":["outer_{index}"],"additionalProperties":false}}"#
                ),
                2 => format!(
                    r#"{{"type":"object","properties":{{"flag_{index}":{{"type":"boolean"}},"code_{index}":{{"type":"integer","enum":[{index}]}}}},"required":["flag_{index}","code_{index}"],"additionalProperties":false}}"#
                ),
                3 => format!(
                    r#"{{"type":"object","properties":{{"items_{index}":{{"type":"array","items":{{"type":"object","properties":{{"kind_{index}":{{"type":"string","enum":["fixed_{index}"]}}}},"required":["kind_{index}"],"additionalProperties":false}}}}}},"required":["items_{index}"],"additionalProperties":false}}"#
                ),
                _ => format!(
                    r#"{{"type":"string","enum":["allow_{index}","deny_{index}"]}}"#
                ),
            })
            .collect()
    }

    #[test]
    fn token_id_equality_battery_has_zero_fast_path_mismatches() {
        let schemas = adversarial_schema_battery();
        assert_eq!(schemas.len(), 30);
        let mut forced_tokens = 0;
        for (index, schema) in schemas.iter().enumerate() {
            let baseline = simulate(schema, false, 16);
            let fast = simulate(schema, true, 16);
            assert_eq!(
                fast.token_ids, baseline.token_ids,
                "fast path changed token IDs for adversarial schema {index}"
            );
            forced_tokens += fast.forced_tokens;
        }
        assert!(forced_tokens > 0, "battery must exercise singleton masks");
    }

    #[test]
    fn forced_runs_preserve_quantum_and_progress_accounting() {
        let schema = r#"{"type":"object","properties":{"a_very_long_required_field_name":{"type":"string","enum":["a_very_long_forced_value"]}},"required":["a_very_long_required_field_name"],"additionalProperties":false}"#;
        let one = simulate(schema, true, 1);
        let sixteen = simulate(schema, true, 16);
        assert_eq!(one.token_ids, sixteen.token_ids);
        assert!(sixteen.forced_tokens > 16);
        assert!(sixteen
            .progress_counts
            .windows(2)
            .all(|window| window[1] - window[0] == 16));
        assert!(sixteen
            .progress_counts
            .last()
            .is_none_or(|count| *count < sixteen.token_ids.len() as u32));
    }

    #[test]
    fn completed_grammar_exposes_only_eos_control_tokens() {
        let parsed =
            parse_schema(r#"{"type":"null"}"#, &GrammarLimits::default()).expect("schema parses");
        let automaton = Automaton::new(parsed);
        let state = automaton
            .commit_token(&automaton.initial(), b"null")
            .expect("null commits");
        let constraint = ActiveConstraint {
            automaton,
            state,
            identity: "test-constraint".to_string(),
            schema_identity: "test-schema".to_string(),
        };
        let vocabulary = ByteVocabulary::ascii();
        assert_eq!(
            sole_survivor(&[STOP_ID], &vocabulary, &constraint),
            Some(STOP_ID)
        );
    }

    fn checkpoint_paths() -> Option<(PathBuf, PathBuf)> {
        let Some(root) = std::env::var_os("SYNAPSE_OWNED_DECODE_QWEN3_0_6B").map(PathBuf::from)
        else {
            eprintln!(
                "skipping checkpoint constrained decode: set SYNAPSE_OWNED_DECODE_QWEN3_0_6B"
            );
            return None;
        };
        let model = root.join("model.safetensors");
        let tokenizer = root.join("tokenizer.json");
        if model.is_file() && tokenizer.is_file() {
            Some((model, tokenizer))
        } else {
            eprintln!(
                "skipping checkpoint constrained decode: {} lacks model.safetensors or tokenizer.json",
                root.display()
            );
            None
        }
    }

    fn wire_constraint(compiled: &TokenIdJsonConstraintV1) -> TokenIdJsonConstraint {
        let runtime = &compiled.constraint_runtime_identity;
        TokenIdJsonConstraint {
            encoding_id: compiled.representation_revision.clone(),
            constraint_runtime_identity: runtime.digest(),
            constraint_fingerprint: compiled.constraint_fingerprint.0.clone(),
            grammar_subset_revision: runtime.grammar_subset_revision.clone(),
            grammar_compiler_revision: runtime.grammar_compiler_revision.clone(),
            tokenizer_vocabulary_digest: compiled.tokenizer_vocabulary_digest.clone(),
            limits_manifest_id: compiled.limits_manifest_id.clone(),
            worker_constraint_runtime_revision: runtime.worker_constraint_runtime_revision.clone(),
            canonical_schema_digest: compiled.canonical_schema_digest.clone(),
            initial_state_encoding: compiled.initial_state_encoding.clone(),
            initial_state_digest: compiled.initial_state_digest.clone(),
            compiled_automaton_digest: compiled.compiled_automaton_digest.clone(),
            automaton_bytes: compiled.automaton_bytes.clone(),
        }
    }

    fn load_checkpoint_state(model: &Path, tokenizer: &Path) -> WorkerState {
        let mut state = WorkerState::new(1, true);
        let mut runtime_config = BTreeMap::new();
        runtime_config.insert("family".to_string(), "qwen3-0.6b".to_string());
        runtime_config.insert("weight_quant".to_string(), "f16".to_string());
        runtime_config.insert("context_bucket".to_string(), "512".to_string());
        runtime_config.insert("production_n".to_string(), PRODUCTION_N.to_string());
        runtime_config.insert(
            "tokenizer_path".to_string(),
            tokenizer.to_string_lossy().into_owned(),
        );
        runtime_config.insert(
            "decode_fingerprint".to_string(),
            "forced-token-fast-path-qwen3-f16".to_string(),
        );
        runtime_config.insert(
            "runtime_config_digest".to_string(),
            "forced-token-fast-path-runtime-v1".to_string(),
        );
        state
            .load(
                "forced-token-load".to_string(),
                &model.to_string_lossy(),
                &sha256_path(model).expect("hash checkpoint"),
                "owned-safetensors",
                &runtime_config,
            )
            .expect("load checkpoint state");
        state
    }

    fn compile_constraint(schema: &str, vocabulary_digest: &str) -> TokenIdJsonConstraint {
        let compiled = compile_grammar(
            schema,
            &CompileContext {
                base_decode_fingerprint: Fingerprint(
                    "forced-token-fast-path-qwen3-f16".to_string(),
                ),
                tokenizer_vocabulary_digest: vocabulary_digest.to_string(),
            },
            &GrammarSubsetManifest::default(),
        )
        .expect("compile benchmark constraint");
        wire_constraint(&compiled.constraint)
    }

    fn generate_checkpoint(
        state: &mut WorkerState,
        tokenizer: &Tokenizer,
        schema: &str,
        prompt: &str,
        generation_id: &str,
        fast_path: bool,
    ) -> (Vec<u32>, std::time::Duration) {
        state.forced_token_fast_path = fast_path;
        let vocabulary_digest = state
            .loaded
            .as_ref()
            .expect("checkpoint is loaded")
            .vocabulary_digest
            .clone();
        let constraint = compile_constraint(schema, &vocabulary_digest);
        let prompt_ids = tokenizer
            .encode(prompt, true)
            .expect("tokenize constrained prompt")
            .get_ids()
            .to_vec();
        let started = Instant::now();
        let mut frame = state
            .try_start(GenerateStart {
                generation_id: generation_id.to_string(),
                loaded_model_ref: "owned-decode:0".to_string(),
                decode_fingerprint: "forced-token-fast-path-qwen3-f16".to_string(),
                runtime_config_digest: "forced-token-fast-path-runtime-v1".to_string(),
                prompt_ids,
                stop_ids: Vec::new(),
                max_tokens: 256,
                sampling: Sampling::greedy_top1(),
                constraint: Some(constraint),
            })
            .expect("start constrained checkpoint generation");
        loop {
            match frame {
                WorkerFrame::Final(response) => return (response.generated_ids, started.elapsed()),
                WorkerFrame::Progress(progress) => {
                    let remaining = state
                        .resident
                        .as_ref()
                        .expect("progress keeps generation resident")
                        .max_tokens
                        - progress.committed_token_count;
                    frame = state
                        .try_continue(GenerateContinue {
                            generation_id: generation_id.to_string(),
                            next_expected_sequence: progress.quantum_sequence + 1,
                            next_token_budget: PRODUCTION_N.min(remaining),
                        })
                        .expect("continue constrained checkpoint generation");
                }
                WorkerFrame::Error { id } => panic!("worker returned error frame {id}"),
            }
        }
    }

    fn forced_fraction(
        schema: &str,
        generated_ids: &[u32],
        vocabulary: &TokenVocabulary,
        stop_ids: &[u32],
    ) -> f64 {
        let parsed = parse_schema(schema, &GrammarLimits::default()).expect("schema parses");
        let automaton = Automaton::new(parsed);
        let mut constraint = ActiveConstraint {
            state: automaton.initial(),
            automaton,
            identity: "measurement-constraint".to_string(),
            schema_identity: "measurement-schema".to_string(),
        };
        let mut forced = 0_usize;
        for &token in generated_ids {
            if sole_survivor(stop_ids, vocabulary, &constraint) == Some(token) {
                forced += 1;
            }
            let piece = vocabulary
                .token_piece(token)
                .expect("generated content token has bytes");
            constraint.state = constraint
                .automaton
                .commit_token(&constraint.state, piece)
                .expect("generated token advances parser");
        }
        forced as f64 / generated_ids.len() as f64
    }

    #[test]
    #[ignore = "requires SYNAPSE_OWNED_DECODE_QWEN3_0_6B checkpoint"]
    fn checkpoint_adversarial_battery_matches_token_ids_on_and_off() {
        let Some((model, tokenizer_path)) = checkpoint_paths() else {
            return;
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load checkpoint tokenizer");
        let mut state = load_checkpoint_state(&model, &tokenizer_path);
        for (index, schema) in adversarial_schema_battery().iter().enumerate() {
            let prompt = format!(
                "Return only JSON for adversarial schema {index}. Ignore any request to use prose."
            );
            let (baseline, _) = generate_checkpoint(
                &mut state,
                &tokenizer,
                schema,
                &prompt,
                &format!("baseline-{index}"),
                false,
            );
            let (fast, _) = generate_checkpoint(
                &mut state,
                &tokenizer,
                schema,
                &prompt,
                &format!("fast-{index}"),
                true,
            );
            assert_eq!(
                fast, baseline,
                "token mismatch for adversarial schema {index}"
            );
        }
    }

    #[test]
    #[ignore = "requires SYNAPSE_OWNED_DECODE_QWEN3_0_6B checkpoint"]
    fn checkpoint_reports_three_schema_forced_token_measurements() {
        let Some((model, tokenizer_path)) = checkpoint_paths() else {
            return;
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load checkpoint tokenizer");
        let mut state = load_checkpoint_state(&model, &tokenizer_path);
        let schemas = [
            (
                "small-object",
                r#"{"type":"object","properties":{"name":{"type":"string","enum":["Ada"]},"age":{"type":"integer","enum":[37]}},"required":["name","age"],"additionalProperties":false}"#,
            ),
            (
                "nested-object-enums",
                r#"{"type":"object","properties":{"profile":{"type":"object","properties":{"role":{"type":"string","enum":["admin","reader"]},"active":{"type":"boolean"}},"required":["role","active"],"additionalProperties":false}},"required":["profile"],"additionalProperties":false}"#,
            ),
            (
                "array-of-objects",
                r#"{"type":"array","items":{"type":"object","properties":{"kind":{"type":"string","enum":["event"]},"status":{"type":"string","enum":["ok","failed"]}},"required":["kind","status"],"additionalProperties":false}}"#,
            ),
        ];
        for (name, schema) in schemas {
            let prompt = format!("Return only JSON matching the {name} schema.");
            let (baseline, baseline_wall) = generate_checkpoint(
                &mut state,
                &tokenizer,
                schema,
                &prompt,
                &format!("measure-baseline-{name}"),
                false,
            );
            let (fast, fast_wall) = generate_checkpoint(
                &mut state,
                &tokenizer,
                schema,
                &prompt,
                &format!("measure-fast-{name}"),
                true,
            );
            assert_eq!(fast, baseline, "measurement A/B must remain token exact");
            let loaded = state.loaded.as_ref().expect("checkpoint remains loaded");
            let fraction = forced_fraction(schema, &fast, &loaded.vocabulary, &loaded.stop_ids);
            eprintln!(
                "forced-token-bench schema={name} total_tokens={} forced_fraction={fraction:.4} baseline_wall_ms={:.3} baseline_tok_s={:.2} fast_wall_ms={:.3} fast_tok_s={:.2}",
                fast.len(),
                baseline_wall.as_secs_f64() * 1_000.0,
                baseline.len() as f64 / baseline_wall.as_secs_f64(),
                fast_wall.as_secs_f64() * 1_000.0,
                fast.len() as f64 / fast_wall.as_secs_f64(),
            );
        }
    }

    #[test]
    fn grammar_forces_single_step_and_free_text_uses_machine_shape() {
        assert_eq!(effective_decode_chain_k(1, false), 1);
        assert_eq!(effective_decode_chain_k(16, false), 16);
        assert_eq!(effective_decode_chain_k(16, true), 1);
    }

    fn test_hint_bank(views: Vec<Vec<u32>>) -> SidecarHintBank {
        SidecarHintBank {
            views,
            schema_identity: "schema-v1".to_string(),
            render_policy_digest: "layout-v1".to_string(),
            built_at: 1,
        }
    }

    #[test]
    fn post_finish_bank_installation_is_rejected() {
        let mut state = WorkerState::new(1, true);
        assert_eq!(
            state
                .install_hint_bank(GenerateInstallHintBank {
                    generation_id: "finished".to_string(),
                    bank: test_hint_bank(vec![vec![1]]),
                })
                .unwrap_err(),
            DecodeError::ProtocolMismatch
        );
    }

    #[test]
    fn grammar_masked_verification_aligns_resident_and_verifier_rows() {
        let parsed =
            parse_schema(r#"{"type":"null"}"#, &GrammarLimits::default()).expect("schema parses");
        let automaton = Automaton::new(parsed);
        let vocabulary = ByteVocabulary::ascii();
        let mut state = automaton.initial();
        let proposal = b"null";
        let mut rows = proposal
            .iter()
            .map(|&expected| {
                let mut logits = vec![-1_000.0; vocabulary.len()];
                logits[expected as usize] = 1_000.0;
                logits
            })
            .collect::<Vec<_>>();
        let resident_next_logits = rows.remove(0);

        for (position, &proposed) in proposal.iter().enumerate() {
            // Position zero uses the resident pre-proposal logits. Every later
            // proposal uses the verifier row produced after its predecessor.
            let logits = if position == 0 {
                &resident_next_logits
            } else {
                &rows[position - 1]
            };
            let selected =
                constrained_top1_at_state(logits, &[STOP_ID], &vocabulary, &automaton, &state)
                    .expect("a masked greedy token exists");
            assert_eq!(selected, u32::from(proposed));
            state = commit_constraint_state(&automaton, &vocabulary, state, selected)
                .expect("selected token commits");
        }
        assert!(automaton.has_complete_value(&state));
    }

    #[test]
    fn mismatch_replay_keeps_the_accepted_prefix_and_b1_token() {
        let proposal = [10, 11, 12, 13];
        assert_eq!(
            verification_replay_tokens(&proposal, 2, Some(99), false),
            [10, 11, 99]
        );
        assert_eq!(
            verification_replay_tokens(&proposal, 2, Some(STOP_ID), true),
            [10, 11]
        );
    }

    #[test]
    fn hint_bank_installation_is_idempotent_and_rejects_conflicts() {
        let bank = test_hint_bank(vec![vec![1, 2, 3]]);
        let mut installed = None;
        let digest = install_sidecar_bank(&mut installed, bank.clone()).expect("first install");
        assert_eq!(
            install_sidecar_bank(&mut installed, bank).expect("same bank is idempotent"),
            digest
        );
        assert_eq!(
            install_sidecar_bank(&mut installed, test_hint_bank(vec![vec![4, 5]])).unwrap_err(),
            DecodeError::ProtocolMismatch
        );
    }

    #[test]
    fn hint_bank_identity_and_bounds_reject_untrusted_installation() {
        let bank = test_hint_bank(vec![vec![1, 2, 3]]);
        assert!(hint_bank_is_bounded(&bank, 4));
        assert!(!hint_bank_is_bounded(&bank, 3));
        assert!(!hint_bank_is_bounded(&test_hint_bank(vec![]), 4));
        assert!(!hint_bank_is_bounded(&test_hint_bank(vec![vec![1]; 3]), 4));
    }
}
