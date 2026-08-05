//! Mandatory `macos-metal` lane certification probe.
//!
//! Compiles only on macOS and is checkpoint-gated (`#[ignore]`): it runs when
//! the model-snapshot and spike-reference environment variables are set, which
//! is how the mandatory `macos-metal` CI lane executes it. The probe drives the
//! real production Metal step engines through the [`DecodeProbe`] seam,
//! registers the reviewed spike reference streams as the independent oracle,
//! and records certification evidence for all four lanes (both families, both
//! formats): byte-identical parity for Q8, structural-band forks for f16.
//!
//! Environment variables (same contract as the engine parity test):
//! - `SYNAPSE_OWNED_DECODE_QWEN3_0_6B`: Qwen3-Embedding-0.6B snapshot directory
//! - `SYNAPSE_OWNED_DECODE_LFM2_1_2B`: LFM2-1.2B snapshot directory
//! - `SYNAPSE_OWNED_DECODE_SPIKE_REFERENCES`: spike reference fixture directory
//!   (defaults to `bench/spikes/unified-rt/fixtures`)
//! - `SYNAPSE_OWNED_DECODE_SPIKE_QWEN3_REFERENCES`: optional override for the
//!   directory holding the pinned Qwen3 spike fixtures (defaults to the spike
//!   fixtures directory; the engine parity test pins the fixture sha256)
//!
//! Run with:
//! ```text
//! SYNAPSE_OWNED_DECODE_QWEN3_0_6B=<qwen3-snapshot> \
//! SYNAPSE_OWNED_DECODE_LFM2_1_2B=<lfm2-snapshot> \
//! SYNAPSE_OWNED_DECODE_SPIKE_REFERENCES=<spike-fixtures-dir> \
//! SYNAPSE_OWNED_DECODE_SPIKE_QWEN3_REFERENCES=<spike-cli-output-dir> \
//! cargo test -p synapse-module --release metal_certification_probe -- --ignored --nocapture
//! ```

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;
use synapse_core::Fingerprint;
use tokenizers::Tokenizer;

use synapse_engine_owned::owned_decode_engine::{
    DecodeKernel, Lfm2DecodeModel, Lfm2HybridStepEngine, MetalStepDecoder, MetalStepKvCache,
    Qwen3DecodeModel, WeightQuantization,
};
use synapse_engine_owned::Precision;

use crate::owned_decode_certification::fixtures::{
    parity_battery, OracleProvenance, OracleStore, ParityFixture, PARITY_PROMPT_COUNT,
};
use crate::owned_decode_certification::probe::{CertificationProbe, DecodeProbe};
use crate::owned_decode_routing::certification::{CertificationStore, StructuralBandChecker};
use crate::owned_decode_routing::family::Family;
use crate::owned_decode_routing::identity::{
    ActivationDType, DecodeIdentityInputs, Q8Identity, WeightQuant,
};

/// One row from the spike's decode-prompts.jsonl fixture.
#[derive(Clone, Deserialize)]
struct DecodePrompt {
    id: String,
    prompt: String,
}

/// One row from a spike LFM2 reference fixture JSONL.
#[derive(Clone, Deserialize)]
struct ReferenceRow {
    id: String,
    tokens: Vec<u32>,
}

/// The spike's Qwen3 CLI output JSON.
#[derive(Clone, Deserialize)]
struct SpikeCliOutput {
    results: Vec<SpikeCliResult>,
}

#[derive(Clone, Deserialize)]
struct SpikeCliResult {
    id: String,
    tokens: Vec<u32>,
}

fn load_decode_prompts() -> Vec<DecodePrompt> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/spikes/unified-rt/decode-prompts.jsonl");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read decode-prompts.jsonl at {}: {error}", path.display()));
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("decode-prompts.jsonl row parses"))
        .collect()
}

fn spike_fixtures_dir() -> PathBuf {
    std::env::var_os("SYNAPSE_OWNED_DECODE_SPIKE_REFERENCES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/spikes/unified-rt/fixtures")
        })
}

/// Load a JSONL reference fixture keyed by prompt ID.
fn load_reference_fixture(name: &str) -> HashMap<String, Vec<u32>> {
    let path = spike_fixtures_dir().join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {name} at {}: {error}", path.display()));
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let row: ReferenceRow = serde_json::from_str(line).expect("reference row parses");
            (row.id, row.tokens)
        })
        .collect()
}

/// Load a pinned spike Qwen3 reference JSON keyed by prompt ID. The fixtures
/// live in the spike fixtures directory; the engine parity test pins their
/// sha256 so the reference cannot drift silently.
fn load_spike_qwen3_output(name: &str) -> HashMap<String, Vec<u32>> {
    let dir = std::env::var_os("SYNAPSE_OWNED_DECODE_SPIKE_QWEN3_REFERENCES")
        .map(PathBuf::from)
        .unwrap_or_else(spike_fixtures_dir);
    let path = dir.join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {name} at {}: {error}", path.display()));
    let output: SpikeCliOutput = serde_json::from_str(&raw).expect("spike CLI output parses");
    output
        .results
        .into_iter()
        .map(|r| (r.id, r.tokens))
        .collect()
}

/// The spike reference file for one fixture lane.
fn spike_reference(fixture: &ParityFixture) -> HashMap<String, Vec<u32>> {
    match (fixture.family, fixture.weight_quant) {
        (Family::Qwen3_0_6b, WeightQuant::F16) => load_spike_qwen3_output("spike-qwen3-f16.jsonl"),
        (Family::Qwen3_0_6b, WeightQuant::Q8_0) => load_spike_qwen3_output("spike-qwen3-q8.jsonl"),
        (Family::Lfm2_1_2b, WeightQuant::F16) => {
            load_reference_fixture("lfm2-f16-step-reference.jsonl")
        }
        (Family::Lfm2_1_2b, WeightQuant::Q8_0) => {
            load_reference_fixture("lfm2-q8-step-reference.jsonl")
        }
    }
}

/// Register the reviewed spike reference streams as the independent oracle for
/// one fixture, keyed by prompt index.
fn register_oracle(oracle: &mut OracleStore, fixture: &ParityFixture, prompts: &[DecodePrompt]) {
    let reference = spike_reference(fixture);
    for (index, prompt) in prompts.iter().enumerate() {
        let tokens = reference
            .get(&prompt.id)
            .unwrap_or_else(|| panic!("spike reference missing prompt {}", prompt.id))
            .clone();
        oracle
            .register(
                &fixture.id,
                index as u32,
                OracleProvenance::SpikeHarness,
                tokens,
            )
            .expect("first oracle registration succeeds");
    }
}

fn fixture_fingerprint(fixture: &ParityFixture) -> Fingerprint {
    DecodeIdentityInputs {
        family: fixture.family,
        activation_dtype: ActivationDType::F16,
        weight_quant: fixture.weight_quant,
        artifact_source_digest: fixture.source_digest.clone(),
        arithmetic_identity_revision: fixture.arithmetic_identity_revision.clone(),
        q8: match (&fixture.q8_quantizer_revision, &fixture.q8_derived_digest) {
            (Some(revision), Some(digest)) => Some(Q8Identity {
                quantizer_revision: revision.clone(),
                derived_digest: digest.clone(),
            }),
            _ => None,
        },
    }
    .decode_fingerprint()
    .expect("fixture identity inputs are valid")
}

/// Qwen3 lane probe: drives the production Metal step decoder. The decoder
/// borrows the model, so both live behind shared references for the probe's
/// lifetime. For Q8 lanes the f16 prefill decoder is present.
struct Qwen3LaneProbe<'a> {
    f16_decoder: Option<MetalStepDecoder<'a>>,
    step_decoder: MetalStepDecoder<'a>,
    model: &'a Qwen3DecodeModel,
    tokenizer: &'a Tokenizer,
    prompt_texts: &'a [String],
    stop_tokens: std::collections::HashSet<u32>,
    max_tokens: usize,
    weight_quant: WeightQuantization,
}

impl DecodeProbe for Qwen3LaneProbe<'_> {
    fn generate(&mut self, _fixture: &ParityFixture, prompt_index: u32) -> Vec<u32> {
        qwen3_greedy_decode(
            &mut self.f16_decoder,
            &mut self.step_decoder,
            self.model,
            self.tokenizer,
            &self.prompt_texts[prompt_index as usize],
            self.max_tokens,
            &self.stop_tokens,
            self.weight_quant,
        )
    }
}

/// LFM2 lane probe: drives the production hybrid step engine. The engine is
/// owned; the model borrow keeps the weight source alive for the probe's
/// lifetime.
struct Lfm2LaneProbe<'a> {
    engine: Lfm2HybridStepEngine,
    _model: &'a Lfm2DecodeModel,
    tokenizer: &'a Tokenizer,
    prompt_texts: &'a [String],
    stop_tokens: std::collections::HashSet<u32>,
    max_tokens: usize,
}

impl DecodeProbe for Lfm2LaneProbe<'_> {
    fn generate(&mut self, _fixture: &ParityFixture, prompt_index: u32) -> Vec<u32> {
        lfm2_greedy_decode(
            &mut self.engine,
            self.tokenizer,
            &self.prompt_texts[prompt_index as usize],
            self.max_tokens,
            &self.stop_tokens,
        )
    }
}

fn load_tokenizer(snapshot: &std::path::Path) -> Tokenizer {
    let mut tokenizer =
        Tokenizer::from_file(snapshot.join("tokenizer.json")).expect("load tokenizer");
    tokenizer.with_padding(None);
    tokenizer.with_truncation(None).expect("disable truncation");
    tokenizer
}

/// Greedy-decode one Qwen3 prompt through the production Metal step engine.
/// For Q8, prefill runs on the f16 engine and the KV cache is imported into the
/// Q8 engine before stepping, matching the spike byte for byte.
fn qwen3_greedy_decode(
    f16_decoder: &mut Option<MetalStepDecoder>,
    step_decoder: &mut MetalStepDecoder,
    model: &Qwen3DecodeModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    stop_tokens: &std::collections::HashSet<u32>,
    weight_quant: WeightQuantization,
) -> Vec<u32> {
    let encoding = tokenizer.encode(prompt, true).expect("encode Qwen3 prompt");
    let prompt_ids = encoding.get_ids().to_vec();
    assert!(!prompt_ids.is_empty(), "Qwen3 prompt produced no tokens");

    let (first, mut cache) = if weight_quant.is_quantized() {
        let f16 = f16_decoder.as_mut().expect("f16 decoder for Q8 prefill");
        let (f16_cache, first) = f16.prefill(&prompt_ids).expect("f16 prefill");
        let one_layer_elements = 2 * model.config.num_key_value_heads * 512 * model.config.head_dim;
        let mut cache_bits = Vec::with_capacity(model.layers.len() * one_layer_elements);
        for layer in 0..model.layers.len() {
            cache_bits.extend(f16.inspect_cache_bits(layer).expect("export f16 cache"));
        }
        step_decoder
            .import_caches(&cache_bits)
            .expect("import caches into Q8");
        (
            first,
            MetalStepKvCache {
                position: f16_cache.position,
            },
        )
    } else {
        let (cache, first) = step_decoder.prefill(&prompt_ids).expect("f16 prefill");
        (first, cache)
    };

    let mut generated = Vec::with_capacity(max_tokens);
    generated.push(first);
    if stop_tokens.contains(&first) || max_tokens <= 1 {
        return generated;
    }
    let remaining = max_tokens - 1;
    let chain_steps = remaining.min(step_decoder.capacity() - cache.position);
    if chain_steps > 0 {
        let tokens = step_decoder
            .advance_chain(&mut cache, first, chain_steps)
            .expect("metal chain");
        for token in tokens {
            generated.push(token);
            if stop_tokens.contains(&token) {
                break;
            }
        }
    }
    generated
}

/// Greedy-decode one LFM2 prompt through the production hybrid step engine.
fn lfm2_greedy_decode(
    engine: &mut Lfm2HybridStepEngine,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    stop_tokens: &std::collections::HashSet<u32>,
) -> Vec<u32> {
    engine.reset().expect("reset caches");
    let encoding = tokenizer.encode(prompt, true).expect("encode LFM2 prompt");
    let prompt_ids = encoding.get_ids().to_vec();
    assert!(!prompt_ids.is_empty(), "LFM2 prompt produced no tokens");

    let first = engine.prefill(&prompt_ids).expect("metal prefill");
    let mut generated = Vec::with_capacity(max_tokens);
    generated.push(first);
    if stop_tokens.contains(&first) || max_tokens <= 1 {
        return generated;
    }
    let position = prompt_ids.len();
    let remaining = max_tokens - 1;
    let chain_steps = remaining.min(engine.capacity() - position);
    if chain_steps > 0 {
        let tokens = engine
            .chain(position, chain_steps, first)
            .expect("metal chain");
        for token in tokens {
            generated.push(token);
            if stop_tokens.contains(&token) {
                break;
            }
        }
    }
    generated
}

/// Run the certification probe through the real Metal engines for all four
/// lanes and record the evidence. Checkpoint-gated: runs only when the model
/// and spike-reference environment variables are set (the mandatory
/// `macos-metal` lane).
#[test]
#[ignore]
fn metal_certification_probe_four_lanes() {
    let manifest_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("owned-decode-manifests");
    let manifests = crate::owned_decode_contracts::load_manifest_dir(&manifest_dir)
        .expect("checked-in manifests load");

    let prompts = load_decode_prompts();
    assert_eq!(
        prompts.len() as u32,
        PARITY_PROMPT_COUNT,
        "the pinned prompt fixture has exactly {PARITY_PROMPT_COUNT} prompts"
    );

    let battery = parity_battery();
    let mut oracle = OracleStore::new();
    for fixture in &battery {
        register_oracle(&mut oracle, fixture, &prompts);
    }

    let checker = StructuralBandChecker::from_manifest(&manifests.structural_band);
    let probe = CertificationProbe::new(
        "metal-lane-profile",
        manifests.fixture_registry.manifest_revision.clone(),
        &oracle,
        checker,
    );
    let mut store = CertificationStore::new();

    let prompt_texts: Vec<String> = prompts.iter().map(|p| p.prompt.clone()).collect();

    for fixture in &battery {
        let fp = fixture_fingerprint(fixture);
        let max_tokens = fixture.max_tokens as usize;
        let evidence = match (fixture.family, fixture.weight_quant) {
            (Family::Qwen3_0_6b, quant) => {
                let snapshot = PathBuf::from(
                    std::env::var_os("SYNAPSE_OWNED_DECODE_QWEN3_0_6B")
                        .expect("set SYNAPSE_OWNED_DECODE_QWEN3_0_6B to the Qwen3 snapshot"),
                );
                let tokenizer = load_tokenizer(&snapshot);
                let weight_quant = match quant {
                    WeightQuant::F16 => WeightQuantization::None,
                    WeightQuant::Q8_0 => WeightQuantization::Q8_0,
                };
                let f16_model = if weight_quant.is_quantized() {
                    Some(
                        Qwen3DecodeModel::load_with_quant(
                            &snapshot.join("model.safetensors"),
                            Precision::F16,
                            WeightQuantization::None,
                        )
                        .expect("load Qwen3 f16 model for Q8 prefill"),
                    )
                } else {
                    None
                };
                let f16_decoder = f16_model.as_ref().map(|model| {
                    MetalStepDecoder::new(model, Precision::F16, 512, WeightQuantization::None)
                        .expect("create Qwen3 f16 prefill decoder")
                });
                let model = Qwen3DecodeModel::load_with_quant(
                    &snapshot.join("model.safetensors"),
                    Precision::F16,
                    weight_quant,
                )
                .expect("load Qwen3 decode model");
                let step_decoder = MetalStepDecoder::new(&model, Precision::F16, 512, weight_quant)
                    .expect("create Qwen3 Metal step decoder");
                let mut lane_probe = Qwen3LaneProbe {
                    f16_decoder,
                    step_decoder,
                    stop_tokens: model.generation_stop_ids().iter().copied().collect(),
                    model: &model,
                    tokenizer: &tokenizer,
                    prompt_texts: &prompt_texts,
                    max_tokens,
                    weight_quant,
                };
                // `f16_model` stays bound in this arm until after `lane_probe`
                // drops, so the prefill decoder's borrow stays valid.
                probe
                    .certify_unconstrained_lane(&mut lane_probe, fixture, fp, &mut store)
                    .unwrap_or_else(|error| {
                        panic!("metal certification failed for {}: {error:?}", fixture.id)
                    })
            }
            (Family::Lfm2_1_2b, quant) => {
                let snapshot = PathBuf::from(
                    std::env::var_os("SYNAPSE_OWNED_DECODE_LFM2_1_2B")
                        .expect("set SYNAPSE_OWNED_DECODE_LFM2_1_2B to the LFM2 snapshot"),
                );
                let tokenizer = load_tokenizer(&snapshot);
                let weight_quant = match quant {
                    WeightQuant::F16 => WeightQuantization::None,
                    WeightQuant::Q8_0 => WeightQuantization::Q8_0,
                };
                let model = Lfm2DecodeModel::load_with_quant(
                    &snapshot.join("model.safetensors"),
                    Precision::F16,
                    weight_quant,
                )
                .expect("load LFM2 decode model");
                let engine = Lfm2HybridStepEngine::new(&model, Precision::F16, 512, weight_quant)
                    .expect("create LFM2 hybrid step engine");
                let mut lane_probe = Lfm2LaneProbe {
                    stop_tokens: model.generation_stop_ids().iter().copied().collect(),
                    engine,
                    _model: &model,
                    tokenizer: &tokenizer,
                    prompt_texts: &prompt_texts,
                    max_tokens,
                };
                probe
                    .certify_unconstrained_lane(&mut lane_probe, fixture, fp, &mut store)
                    .unwrap_or_else(|error| {
                        panic!("metal certification failed for {}: {error:?}", fixture.id)
                    })
            }
        };
        if fixture.weight_quant == WeightQuant::Q8_0 {
            assert_eq!(
                evidence.top2_swaps, 0,
                "Q8 certification requires zero forks on {}",
                fixture.id
            );
        } else {
            assert!(
                evidence.top2_swaps <= 2,
                "f16 certification allows at most two top-2 swaps on {}",
                fixture.id
            );
        }
        println!(
            "[certification] {}: evidence={} top2_swaps={} signature={}",
            fixture.id,
            evidence.evidence_id(),
            evidence.top2_swaps,
            evidence.fork_signature
        );
    }
}
