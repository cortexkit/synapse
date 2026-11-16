//! Test-only spike-harness parity and throughput runners.
//!
//! This module compiles only under `cfg(test)`: it is certification tooling,
//! never linked into the serving path. The parity runner drives a spike-side
//! probe and a production-side probe consecutively over identical
//! registry-defined inputs and compares their token streams byte for byte. The
//! throughput runner drives the same pair under fixed per-token costs and
//! reports the steady-state production/spike ratio with startup excluded and
//! reported separately, mirroring the G-DEC-04 and G-DEC-10 measurement
//! protocol without Metal hardware.

use std::time::Duration;

use crate::owned_decode_certification::fixtures::{
    parity_battery, reference_chain_step, reference_stream_seed, OracleStore, ParityFixture,
};
use crate::owned_decode_certification::probe::{compare_streams, DecodeProbe, ForkDivergence};

/// The spike side of the A/B comparison: serves the reviewed reference bytes
/// registered in the oracle store.
pub struct SpikeReferenceProbe<'a> {
    oracle: &'a OracleStore,
}

impl<'a> SpikeReferenceProbe<'a> {
    pub fn new(oracle: &'a OracleStore) -> Self {
        Self { oracle }
    }
}

impl DecodeProbe for SpikeReferenceProbe<'_> {
    fn generate(&mut self, fixture: &ParityFixture, prompt_index: u32) -> Vec<u32> {
        self.oracle
            .stream(&fixture.id, prompt_index)
            .unwrap_or_else(|| panic!("oracle missing {}:{}", fixture.id, prompt_index))
            .to_vec()
    }
}

/// A production-side double that computes the reference hash chain in chunks,
/// modeling chunked (quantum-paced) execution against the oracle's
/// uninterrupted chain. The arithmetic and accumulation order are unchanged,
/// so the streams must be byte-identical; any divergence is a parity failure.
pub struct ChunkedProductionProbe {
    chunk_size: u32,
}

impl ChunkedProductionProbe {
    pub fn new(chunk_size: u32) -> Self {
        assert!(chunk_size > 0, "chunk size must be positive");
        Self { chunk_size }
    }
}

impl DecodeProbe for ChunkedProductionProbe {
    fn generate(&mut self, fixture: &ParityFixture, prompt_index: u32) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(fixture.max_tokens as usize);
        let mut state = reference_stream_seed(&fixture.id, prompt_index);
        let mut remaining = fixture.max_tokens;
        while remaining > 0 {
            let steps = remaining.min(self.chunk_size);
            for _ in 0..steps {
                let (next, token) = reference_chain_step(state);
                state = next;
                tokens.push(token);
            }
            remaining -= steps;
            // A chunk boundary changes pacing only; the arithmetic state
            // carries straight through, exactly like a quantum boundary that
            // releases and reacquires the permit without touching resident
            // cache state.
        }
        tokens
    }
}

/// One production-side fork used by the negative parity test.
pub struct FlippingProductionProbe {
    /// Flip the first token of this prompt index.
    pub flip_prompt: u32,
}

impl DecodeProbe for FlippingProductionProbe {
    fn generate(&mut self, fixture: &ParityFixture, prompt_index: u32) -> Vec<u32> {
        let mut state = reference_stream_seed(&fixture.id, prompt_index);
        let mut tokens = Vec::with_capacity(fixture.max_tokens as usize);
        for _ in 0..fixture.max_tokens {
            let (next, token) = reference_chain_step(state);
            state = next;
            tokens.push(token);
        }
        if prompt_index == self.flip_prompt {
            tokens[0] = tokens[0].wrapping_add(1);
        }
        tokens
    }
}

/// The outcome of one spike-vs-production parity run.
#[derive(Clone, Debug, PartialEq)]
pub struct SpikeParityReport {
    pub fixture_id: String,
    pub prompts_compared: u32,
    pub byte_identical: u32,
    pub first_divergence: Option<ForkDivergence>,
}

impl SpikeParityReport {
    pub fn is_byte_identical(&self) -> bool {
        self.first_divergence.is_none() && self.byte_identical == self.prompts_compared
    }
}

/// Run spike and production consecutively over identical fixture inputs and
/// compare the generated-token-ID streams byte for byte.
pub fn run_spike_parity(
    spike: &mut dyn DecodeProbe,
    production: &mut dyn DecodeProbe,
    fixture: &ParityFixture,
) -> SpikeParityReport {
    let mut byte_identical = 0;
    let mut first_divergence = None;
    for prompt_index in 0..fixture.prompt_count {
        let spike_tokens = spike.generate(fixture, prompt_index);
        let production_tokens = production.generate(fixture, prompt_index);
        let divergences = compare_streams(&production_tokens, &spike_tokens, prompt_index);
        if divergences.is_empty() {
            byte_identical += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(divergences[0].clone());
        }
    }
    SpikeParityReport {
        fixture_id: fixture.id.clone(),
        prompts_compared: fixture.prompt_count,
        byte_identical,
        first_divergence,
    }
}

/// A probe that reports a synthetic per-run duration alongside its tokens, so
/// throughput accounting is deterministic and hardware-independent.
pub trait TimedDecodeProbe {
    fn generate_timed(
        &mut self,
        fixture: &ParityFixture,
        prompt_index: u32,
    ) -> (Vec<u32>, Duration);
}

/// A fixed-rate timed probe: every token costs `per_token`, and the first call
/// additionally pays `startup_overhead` (first load / first ingest).
pub struct FixedRateProbe {
    pub per_token: Duration,
    pub startup_overhead: Duration,
    first_call: bool,
}

impl FixedRateProbe {
    pub fn new(per_token: Duration, startup_overhead: Duration) -> Self {
        Self {
            per_token,
            startup_overhead,
            first_call: true,
        }
    }
}

impl TimedDecodeProbe for FixedRateProbe {
    fn generate_timed(
        &mut self,
        fixture: &ParityFixture,
        prompt_index: u32,
    ) -> (Vec<u32>, Duration) {
        let tokens = vec![0u32; fixture.max_tokens as usize];
        let mut elapsed = self.per_token * fixture.max_tokens;
        if self.first_call {
            elapsed += self.startup_overhead;
            self.first_call = false;
        }
        let _ = prompt_index;
        (tokens, elapsed)
    }
}

/// The outcome of one consecutive spike/production throughput comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct ThroughputRunReport {
    pub fixture_id: String,
    pub spike_tokens_per_sec: f64,
    pub production_tokens_per_sec: f64,
    pub ratio: f64,
    pub warmup_repetitions: u32,
    /// The first production run (startup, first load, first Q8 ingest), timed
    /// and reported separately from steady state.
    pub production_startup: Duration,
}

/// Run spike and production consecutively under identical inputs and compute
/// the steady-state throughput ratio. Warmup repetitions are excluded from the
/// steady-state totals, and the first production run is reported separately as
/// startup.
pub fn run_throughput_comparison(
    spike: &mut dyn TimedDecodeProbe,
    production: &mut dyn TimedDecodeProbe,
    fixture: &ParityFixture,
    warmup: u32,
    repetitions: u32,
) -> ThroughputRunReport {
    let total = warmup + repetitions;
    let mut spike_tokens = 0u64;
    let mut spike_time = Duration::ZERO;
    let mut production_tokens = 0u64;
    let mut production_time = Duration::ZERO;
    let mut production_startup = Duration::ZERO;

    for repetition in 0..total {
        let prompt_index = repetition % fixture.prompt_count;
        let (spike_stream, spike_elapsed) = spike.generate_timed(fixture, prompt_index);
        let (production_stream, production_elapsed) =
            production.generate_timed(fixture, prompt_index);
        if repetition == 0 {
            production_startup = production_elapsed;
        }
        if repetition >= warmup {
            spike_tokens += spike_stream.len() as u64;
            spike_time += spike_elapsed;
            production_tokens += production_stream.len() as u64;
            production_time += production_elapsed;
        }
    }

    let spike_tokens_per_sec = spike_tokens as f64 / spike_time.as_secs_f64();
    let production_tokens_per_sec = production_tokens as f64 / production_time.as_secs_f64();
    ThroughputRunReport {
        fixture_id: fixture.id.clone(),
        spike_tokens_per_sec,
        production_tokens_per_sec,
        ratio: production_tokens_per_sec / spike_tokens_per_sec,
        warmup_repetitions: warmup,
        production_startup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_certification::fixtures::spike_reference_stream;
    use crate::owned_decode_certification::gates::THROUGHPUT_RATIO_BOUND;
    use crate::owned_decode_routing::identity::WeightQuant;

    fn oracle_with_battery() -> (Vec<ParityFixture>, OracleStore) {
        let battery = parity_battery();
        let mut oracle = OracleStore::new();
        oracle.register_synthetic_battery(&battery);
        (battery, oracle)
    }

    #[test]
    fn chunked_production_is_byte_identical_to_the_spike_for_all_four_lanes() {
        let (battery, oracle) = oracle_with_battery();
        let mut families = std::collections::BTreeSet::new();
        let mut formats = std::collections::BTreeSet::new();
        for fixture in &battery {
            let mut spike = SpikeReferenceProbe::new(&oracle);
            // Chunk size 8 crosses at least two quantum boundaries at 64 tokens.
            let mut production = ChunkedProductionProbe::new(8);
            let report = run_spike_parity(&mut spike, &mut production, fixture);
            assert!(
                report.is_byte_identical(),
                "{} must be byte-identical: {:?}",
                fixture.id,
                report
            );
            families.insert(fixture.family.as_str());
            formats.insert(fixture.weight_quant.as_str());
        }
        // Direct parity covers both families and both formats.
        assert_eq!(families.len(), 2);
        assert_eq!(formats.len(), 2);
    }

    #[test]
    fn chunk_boundaries_do_not_change_the_stream() {
        let (battery, _) = oracle_with_battery();
        let fixture = &battery[0];
        let expected = spike_reference_stream(fixture, 3);
        for chunk_size in [1u32, 7, 8, 16, 32, 64] {
            let mut probe = ChunkedProductionProbe::new(chunk_size);
            assert_eq!(
                probe.generate(fixture, 3),
                expected,
                "chunk size {chunk_size} must reproduce the uninterrupted stream"
            );
        }
    }

    #[test]
    fn parity_runner_reports_the_first_divergence() {
        let (battery, oracle) = oracle_with_battery();
        let fixture = &battery[0];
        let mut spike = SpikeReferenceProbe::new(&oracle);
        let mut production = FlippingProductionProbe { flip_prompt: 2 };
        let report = run_spike_parity(&mut spike, &mut production, fixture);
        assert!(!report.is_byte_identical());
        assert_eq!(report.byte_identical, report.prompts_compared - 1);
        let divergence = report.first_divergence.expect("divergence reported");
        assert_eq!(divergence.prompt_index, 2);
        assert_eq!(divergence.step, 0);
    }

    #[test]
    fn throughput_runner_excludes_warmup_and_reports_startup_separately() {
        let (battery, _) = oracle_with_battery();
        let fixture = &battery[0];
        // Spike: 100 us/token. Production: 105 us/token steady state plus a
        // 50 ms first-load startup. Ratio = 100/105 ~= 0.952 >= 0.90.
        let mut spike = FixedRateProbe::new(Duration::from_micros(100), Duration::ZERO);
        let mut production =
            FixedRateProbe::new(Duration::from_micros(105), Duration::from_millis(50));
        let report = run_throughput_comparison(&mut spike, &mut production, fixture, 5, 20);

        assert_eq!(report.warmup_repetitions, 5);
        assert!(
            report.ratio >= THROUGHPUT_RATIO_BOUND,
            "steady-state ratio {} must clear {THROUGHPUT_RATIO_BOUND}",
            report.ratio
        );
        assert!((report.spike_tokens_per_sec - 10_000.0).abs() < 1.0);
        assert!((report.production_tokens_per_sec - 10_000.0 * 100.0 / 105.0).abs() < 1.0);
        // Startup carries the first-load overhead and is separate from the
        // steady-state per-token cost.
        assert_eq!(
            report.production_startup,
            Duration::from_micros(105 * 64) + Duration::from_millis(50)
        );
    }

    #[test]
    fn throughput_runner_surfaces_slow_production() {
        let (battery, _) = oracle_with_battery();
        let fixture = battery
            .iter()
            .find(|f| f.weight_quant == WeightQuant::Q8_0)
            .expect("q8 lane present");
        let mut spike = FixedRateProbe::new(Duration::from_micros(100), Duration::ZERO);
        let mut production = FixedRateProbe::new(Duration::from_micros(150), Duration::ZERO);
        let report = run_throughput_comparison(&mut spike, &mut production, fixture, 2, 10);
        assert!(
            report.ratio < THROUGHPUT_RATIO_BOUND,
            "150 us/token production must fall below the bound"
        );
    }
}
