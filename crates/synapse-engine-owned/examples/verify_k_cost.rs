//! What does it cost to verify K guessed tokens in one pass on the owned Metal
//! decode engine?
//!
//! Speculative decoding guesses K tokens cheaply and then checks all K with
//! the full model. It only pays if checking K tokens in one forward pass costs
//! much less than K single-token steps. This harness measures that cost curve
//! for Qwen3-0.6B on the production decode engine
//! (`owned-decode-engine`, `MetalStepDecoder::verify_tokens_batch`), which runs
//! up to 16 tokens through the transformer in one command buffer: the heavy
//! projections (QKV, O, gate/up, down, LM head) are mat-mat kernels that
//! stream each weight row once for all K columns, while the per-token kernels
//! (norms, RoPE, attention, argmax) are dispatched once per column.
//!
//! For each weight lane (f16 and Q8_0; both are supported by the batched
//! path) and for two KV-cache depths (32 and 470 tokens already cached), it
//! reports the wall time of one `verify_tokens_batch` call for
//! K in {1, 2, 4, 8, 16}, and the ratio cost(K) / cost(1). Attention is the
//! only work whose cost grows with cache depth, so comparing the two depths
//! splits K's extra cost into a depth-dependent part (attention) and a flat
//! part (weight projections plus fixed per-dispatch cost). The split cannot
//! name individual kernels. The engine's own per-kernel profiler
//! (`SYNAPSE_METAL_STEP_PROFILE`) only instruments the single-token `advance`
//! path, and its counters are not reachable from outside the engine, so it
//! cannot be used here. The `trace` mode below runs the verify calls in
//! isolated phases so that an external Metal System Trace (`xctrace`) can
//! measure GPU time per verify call without instrumenting the engine.
//!
//! As context it also times the unbatched reference: `DecodeKernel::verify_tokens`,
//! which encodes K ordinary single-token forward passes into one command
//! buffer. That is what K single-token steps cost with no host round trips
//! between them.
//!
//! Method:
//! - Correctness guard first. At each depth the harness generates a 16-token
//!   greedy continuation with 16 sequential single-token `advance` calls and
//!   records each step's logits. That continuation is the draft. For every
//!   measured K, `verify_tokens_batch_logits` over the first K draft tokens
//!   must reproduce the sequential logits bit for bit and the same greedy
//!   tokens; otherwise the run stops, because timing a wrong answer is
//!   worthless. Every timed call re-checks its returned tokens as well.
//! - Warm-up calls for every (arm, K) before timing.
//! - Arms and K values are interleaved within each round, and the order is
//!   rotated every round, so ambient load on this shared machine hits every K
//!   alike. The ratio is the result; the absolute times are context. Each
//!   round also yields a paired ratio t(K) / t(K=1) from the same round.
//! - Samples are reported as median, p10, p90, min and max (nearest rank).
//!   The 1/5/15-minute load averages and the `uptime` line are recorded at the
//!   start and end of every block of rounds.
//!
//! Every timed call first rewinds the cache to the chosen depth, then verifies
//! the same draft, so each call rewrites the same KV slots with the same
//! values and the cache state is identical across calls.
//!
//! Env vars:
//! - `SYNAPSE_OWNED_DECODE_QWEN3_0_6B` (required): Qwen3 0.6B snapshot
//!   directory holding `model.safetensors`, `config.json` and
//!   `tokenizer.json` (same variable as the other owned-decode harnesses).
//! - `SYNAPSE_VERIFY_K_MODE`: `full` (default), `smoke` (K in {1, 8}, a few
//!   rounds; proves the harness works, its numbers are not a result), or
//!   `trace` (batched K in {1, 8} only, one uninterrupted phase per
//!   (depth, K) with two idle seconds around it, meant to run under
//!   `xctrace record --template 'Metal System Trace'`; see
//!   `docs/evidence/verify-k-cost-m5/`).
//! - `SYNAPSE_VERIFY_K_LANES`: comma list of `f16`, `q8` (default `f16,q8`).
//! - `SYNAPSE_VERIFY_K_ROUNDS`: override the number of timed rounds (or, in
//!   trace mode, calls per phase).
//! - `SYNAPSE_VERIFY_K_OUT`: optional path that also receives the JSON report
//!   (the report is always printed to stdout).
//!
//! Run with:
//! ```text
//! DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
//! SYNAPSE_OWNED_DECODE_QWEN3_0_6B=<qwen3-0.6b-snapshot> \
//! cargo run -p synapse-engine-owned --release --example verify_k_cost
//! ```

#[cfg(target_os = "macos")]
mod imp {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{Duration, Instant};

    use anyhow::{bail, ensure, Context, Result};
    use serde::Serialize;
    use synapse_engine_owned::owned_decode_engine::{
        DecodeKernel, MetalStepDecoder, MetalStepKvCache, Qwen3DecodeModel, WeightQuantization,
    };
    use synapse_engine_owned::Precision;
    use tokenizers::Tokenizer;

    /// Cache depths (tokens already in the KV cache when the verify runs).
    const DEPTHS: [usize; 2] = [32, 470];
    /// Longest draft the batched path accepts.
    const MAX_K: usize = MetalStepDecoder::MAX_BATCH_VERIFY_TOKENS;
    /// Context bucket: the production workload bucket used by the other
    /// owned-decode harnesses. 470 + 16 positions fit comfortably.
    const BUCKET: usize = 2048;
    /// Prefill runs in batched chunks of this many tokens (the production
    /// prefill chunk size); only the resulting KV state matters here.
    const PREFILL_CHUNK: usize = 16;
    /// Plain English text, tokenized and cycled to the longest depth.
    const PROMPT_TEXT: &str = "The harbour town woke slowly. Fishing boats came in \
        with the tide, gulls argued over the nets, and the baker on the corner \
        opened his shutters to the smell of bread. Down by the quay, an old \
        engineer explained to anyone who would listen how the lighthouse lens \
        had been ground by hand, one careful pass at a time, over a whole winter.";

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Full,
        Smoke,
        Trace,
    }

    struct Protocol {
        mode: Mode,
        ks: Vec<usize>,
        warmup: usize,
        rounds: usize,
        block_rounds: usize,
        sequential_arm: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Arm {
        /// `verify_tokens_batch`: K tokens in one batched forward pass.
        Batch,
        /// `DecodeKernel::verify_tokens`: K single-token forward passes
        /// encoded into one command buffer (unbatched reference).
        Sequential,
    }

    #[derive(Serialize)]
    struct Load {
        loadavg_1_5_15: [f64; 3],
        uptime: String,
    }

    #[derive(Serialize)]
    struct Block {
        first_round: usize,
        rounds: usize,
        load_start: Load,
        load_end: Load,
    }

    #[derive(Serialize)]
    struct Stats {
        samples: usize,
        median_ms: f64,
        p10_ms: f64,
        p90_ms: f64,
        min_ms: f64,
        max_ms: f64,
    }

    #[derive(Serialize)]
    struct Cell {
        arm: Arm,
        k: usize,
        wall: Stats,
        /// median(K) / median(K=1 batch) at the same depth and lane.
        ratio_of_medians: f64,
        /// Per-round t(K) / t(K=1 batch) from the same round, summarized
        /// (the `_ms` field names hold unitless ratios here). Absent in trace
        /// mode, whose phases are not interleaved.
        paired_ratio: Option<Stats>,
    }

    #[derive(Serialize)]
    struct Guard {
        k: usize,
        batch_tokens: Vec<u32>,
        sequential_tokens: Vec<u32>,
        logits_bit_identical: bool,
    }

    #[derive(Serialize)]
    struct DepthReport {
        depth: usize,
        draft: Vec<u32>,
        guard: Vec<Guard>,
        blocks: Vec<Block>,
        cells: Vec<Cell>,
    }

    /// Per K: how K's extra batched cost over K=1 changes between the short
    /// and the long cache depth.
    #[derive(Serialize)]
    struct Decomposition {
        k: usize,
        extra_ms_short: f64,
        extra_ms_long: f64,
        /// `extra_ms_long - extra_ms_short`: the part of K's extra cost that
        /// grows with cache depth. Only attention scans the cache, so this is
        /// attention's share of the extra cost added between the two depths
        /// (a lower bound on attention's share at the long depth, because the
        /// short depth already has 32 positions of attention in it).
        depth_dependent_extra_ms: f64,
        /// `extra_ms_short`: the part of K's extra cost that is present at
        /// the short depth. Mostly weight projections and fixed per-dispatch
        /// cost, plus the (small) attention over 32 positions.
        flat_extra_ms: f64,
        /// `depth_dependent_extra_ms / extra_ms_long`.
        depth_dependent_fraction_at_long: f64,
    }

    #[derive(Serialize)]
    struct LaneReport {
        lane: &'static str,
        depths: Vec<DepthReport>,
        decomposition: Vec<Decomposition>,
    }

    #[derive(Serialize)]
    struct Report {
        harness: &'static str,
        mode: &'static str,
        machine: String,
        model_dir: String,
        bucket: usize,
        ks: Vec<usize>,
        warmup_calls_per_cell: usize,
        rounds: usize,
        block_rounds: usize,
        metal_step_profile_env: Option<String>,
        load_at_start: Load,
        lanes: Vec<LaneReport>,
        load_at_end: Load,
    }

    pub fn main() {
        if let Err(error) = run() {
            eprintln!("[verify-k] FAILED: {error:#}");
            std::process::exit(1);
        }
    }

    fn protocol() -> Result<Protocol> {
        let mode = match std::env::var("SYNAPSE_VERIFY_K_MODE")
            .unwrap_or_else(|_| "full".into())
            .as_str()
        {
            "full" => Mode::Full,
            "smoke" => Mode::Smoke,
            "trace" => Mode::Trace,
            other => bail!("SYNAPSE_VERIFY_K_MODE must be full, smoke or trace, got {other}"),
        };
        let mut protocol = match mode {
            Mode::Full => Protocol {
                mode,
                ks: vec![1, 2, 4, 8, 16],
                warmup: 20,
                rounds: 300,
                block_rounds: 50,
                sequential_arm: true,
            },
            Mode::Smoke => Protocol {
                mode,
                ks: vec![1, 8],
                warmup: 3,
                rounds: 12,
                block_rounds: 6,
                sequential_arm: true,
            },
            Mode::Trace => Protocol {
                mode,
                ks: vec![1, 8],
                warmup: 10,
                rounds: 200,
                block_rounds: 200,
                sequential_arm: false,
            },
        };
        if let Some(rounds) = std::env::var("SYNAPSE_VERIFY_K_ROUNDS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
        {
            protocol.rounds = rounds;
            protocol.block_rounds = protocol.block_rounds.min(rounds);
        }
        Ok(protocol)
    }

    fn lanes() -> Result<Vec<WeightQuantization>> {
        let raw = std::env::var("SYNAPSE_VERIFY_K_LANES").unwrap_or_else(|_| "f16,q8".into());
        raw.split(',')
            .map(str::trim)
            .filter(|lane| !lane.is_empty())
            .map(|lane| match lane {
                "f16" => Ok(WeightQuantization::None),
                "q8" => Ok(WeightQuantization::Q8_0),
                other => bail!("SYNAPSE_VERIFY_K_LANES entries must be f16 or q8, got {other}"),
            })
            .collect()
    }

    fn lane_name(quant: WeightQuantization) -> &'static str {
        match quant {
            WeightQuantization::None => "f16",
            WeightQuantization::Q8_0 => "q8_0",
        }
    }

    fn run() -> Result<()> {
        let protocol = protocol()?;
        let lanes = lanes()?;
        let model_dir = PathBuf::from(
            std::env::var_os("SYNAPSE_OWNED_DECODE_QWEN3_0_6B")
                .context("set SYNAPSE_OWNED_DECODE_QWEN3_0_6B to the Qwen3 0.6B snapshot")?,
        );
        let mode_name = match protocol.mode {
            Mode::Full => "full",
            Mode::Smoke => "smoke",
            Mode::Trace => "trace",
        };
        let machine = machine();
        println!(
            "[verify-k] mode={mode_name} model={} machine={machine}",
            model_dir.display()
        );
        if protocol.mode == Mode::Smoke {
            println!("[verify-k] SMOKE PASS: proves the harness runs; numbers are not a result");
        }
        let load_at_start = load();
        println!(
            "[verify-k] load at start: {}",
            load_at_start.uptime.trim_end()
        );

        let prompt = prompt_tokens(&model_dir)?;
        let started = Instant::now();
        let mut lane_reports = Vec::new();
        for quant in lanes {
            lane_reports.push(run_lane(&model_dir, quant, &prompt, &protocol, started)?);
        }

        let report = Report {
            harness: "verify_k_cost",
            mode: mode_name,
            machine,
            model_dir: model_dir.display().to_string(),
            bucket: BUCKET,
            ks: protocol.ks.clone(),
            warmup_calls_per_cell: protocol.warmup,
            rounds: protocol.rounds,
            block_rounds: protocol.block_rounds,
            metal_step_profile_env: std::env::var("SYNAPSE_METAL_STEP_PROFILE").ok(),
            load_at_start,
            lanes: lane_reports,
            load_at_end: load(),
        };
        print_summary(&report);
        let json = serde_json::to_string_pretty(&report)?;
        println!("{json}");
        if let Some(out) = std::env::var_os("SYNAPSE_VERIFY_K_OUT") {
            std::fs::write(&out, &json)
                .with_context(|| format!("write report to {}", Path::new(&out).display()))?;
        }
        Ok(())
    }

    fn prompt_tokens(model_dir: &Path) -> Result<Vec<u32>> {
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
        let base = tokenizer
            .encode(PROMPT_TEXT, false)
            .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?
            .get_ids()
            .to_vec();
        ensure!(!base.is_empty(), "prompt produced no tokens");
        let longest = *DEPTHS.iter().max().expect("depths");
        Ok((0..longest).map(|index| base[index % base.len()]).collect())
    }

    fn run_lane(
        model_dir: &Path,
        quant: WeightQuantization,
        prompt: &[u32],
        protocol: &Protocol,
        started: Instant,
    ) -> Result<LaneReport> {
        let lane = lane_name(quant);
        println!("[verify-k] lane {lane}: loading model");
        let model = Qwen3DecodeModel::load_with_quant(
            &model_dir.join("model.safetensors"),
            Precision::F16,
            quant,
        )?;
        let mut decoder = MetalStepDecoder::new(model, Precision::F16, BUCKET, quant)?;

        let mut depths = Vec::new();
        for depth in DEPTHS {
            let (mut cache, first) = prefill(&mut decoder, &prompt[..depth])?;
            let (draft, expected, guard) =
                correctness_guard(&mut decoder, &mut cache, depth, first, &protocol.ks)?;
            println!(
                "[verify-k] lane {lane} depth {depth}: guard passed (batched == sequential, \
                 logits bit-identical) for K in {:?}",
                protocol.ks
            );
            let bench = Bench {
                decoder: &mut decoder,
                cache: &mut cache,
                depth,
                draft: &draft,
                expected: &expected,
                lane,
            };
            let (blocks, cells) = if protocol.mode == Mode::Trace {
                trace_phases(bench, protocol, started)?
            } else {
                timed_rounds(bench, protocol)?
            };
            depths.push(DepthReport {
                depth,
                draft,
                guard,
                blocks,
                cells,
            });
        }
        let decomposition = decompose(&depths, &protocol.ks);
        Ok(LaneReport {
            lane,
            depths,
            decomposition,
        })
    }

    /// Fill the KV cache with `tokens` through batched prefill chunks. Returns
    /// the cache positioned right after them and the greedy token that follows
    /// the last one.
    fn prefill(decoder: &mut MetalStepDecoder, tokens: &[u32]) -> Result<(MetalStepKvCache, u32)> {
        let mut cache = MetalStepKvCache { position: 0 };
        let mut first = 0;
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            first = *decoder
                .verify_tokens_batch(&mut cache, chunk)?
                .last()
                .context("empty prefill chunk")?;
        }
        ensure!(
            cache.position == tokens.len(),
            "prefill advanced to {}, expected {}",
            cache.position,
            tokens.len()
        );
        Ok((cache, first))
    }

    fn argmax(logits: &[f32]) -> u32 {
        // First maximum wins on ties.
        let mut best = 0usize;
        for (index, &value) in logits.iter().enumerate() {
            if value > logits[best] {
                best = index;
            }
        }
        best as u32
    }

    /// Builds the draft and proves the batched verify agrees with sequential
    /// single-token steps before anything is timed.
    ///
    /// The draft is the model's own greedy continuation, so every verify is a
    /// full accept (the case speculative decoding hopes for). Starting from
    /// `first`, the greedy token after the prompt, 16 sequential single-token
    /// `advance` calls produce the next 16 greedy tokens and their logits.
    /// `draft[i]` is the token fed at position depth+i and `expected[i]` is the
    /// greedy token after it. For each K, the batched verify of `draft[..K]`
    /// must return `expected[..K]` and bit-identical logits.
    fn correctness_guard(
        decoder: &mut MetalStepDecoder,
        cache: &mut MetalStepKvCache,
        depth: usize,
        first: u32,
        ks: &[usize],
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<Guard>)> {
        let mut draft = Vec::with_capacity(MAX_K);
        let mut expected = Vec::with_capacity(MAX_K);
        let mut sequential_logits = Vec::with_capacity(MAX_K);
        let mut token = first;
        for _ in 0..MAX_K {
            draft.push(token);
            let logits = DecodeKernel::advance(decoder, cache, token)?;
            token = argmax(&logits);
            expected.push(token);
            sequential_logits.push(logits);
        }

        let mut guards = Vec::new();
        for &k in ks {
            DecodeKernel::rewind(decoder, cache, depth)?;
            let batch_logits = decoder.verify_tokens_batch_logits(cache, &draft[..k])?;
            let logits_bit_identical = batch_logits
                .chunks(batch_logits.len() / k)
                .zip(&sequential_logits)
                .all(|(batch, sequential)| {
                    batch.len() == sequential.len()
                        && batch
                            .iter()
                            .zip(sequential)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                });
            DecodeKernel::rewind(decoder, cache, depth)?;
            let batch_tokens = decoder.verify_tokens_batch(cache, &draft[..k])?;
            let sequential_tokens = expected[..k].to_vec();
            ensure!(
                logits_bit_identical && batch_tokens == sequential_tokens,
                "correctness guard FAILED at depth {depth} K={k}: batched verify returned \
                 {batch_tokens:?}, sequential steps gave {sequential_tokens:?}, \
                 logits bit-identical: {logits_bit_identical}. Not timing a wrong answer."
            );
            guards.push(Guard {
                k,
                batch_tokens,
                sequential_tokens,
                logits_bit_identical,
            });
        }
        DecodeKernel::rewind(decoder, cache, depth)?;
        Ok((draft, expected, guards))
    }

    /// Everything one timed call needs.
    struct Bench<'a> {
        decoder: &'a mut MetalStepDecoder,
        cache: &'a mut MetalStepKvCache,
        depth: usize,
        draft: &'a [u32],
        expected: &'a [u32],
        lane: &'static str,
    }

    impl Bench<'_> {
        /// One timed verify of the first `k` draft tokens at the fixed depth.
        /// Returns wall milliseconds for the verify call alone (the rewind is
        /// a host-side bookkeeping change and is outside the timed span).
        fn call(&mut self, arm: Arm, k: usize) -> Result<f64> {
            DecodeKernel::rewind(self.decoder, self.cache, self.depth)?;
            let draft = &self.draft[..k];
            let started = Instant::now();
            let tokens = match arm {
                Arm::Batch => self.decoder.verify_tokens_batch(self.cache, draft)?,
                Arm::Sequential => DecodeKernel::verify_tokens(self.decoder, self.cache, draft)?,
            };
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            ensure!(
                tokens == self.expected[..k],
                "timed {arm:?} K={k} at depth {} returned {tokens:?}, expected {:?}",
                self.depth,
                &self.expected[..k]
            );
            Ok(elapsed)
        }
    }

    fn arms(protocol: &Protocol) -> Vec<(Arm, usize)> {
        let mut arms: Vec<(Arm, usize)> = protocol.ks.iter().map(|&k| (Arm::Batch, k)).collect();
        if protocol.sequential_arm {
            arms.extend(protocol.ks.iter().map(|&k| (Arm::Sequential, k)));
        }
        arms
    }

    /// Interleaved rounds: every round times each (arm, K) once, in an order
    /// rotated by one each round, so slow ambient periods land on all K alike.
    fn timed_rounds(mut bench: Bench<'_>, protocol: &Protocol) -> Result<(Vec<Block>, Vec<Cell>)> {
        let arms = arms(protocol);
        let baseline = arms
            .iter()
            .position(|&arm| arm == (Arm::Batch, 1))
            .context("K=1 must be measured: it is the ratio denominator")?;
        for _ in 0..protocol.warmup {
            for &(arm, k) in &arms {
                bench.call(arm, k)?;
            }
        }
        let mut samples = vec![Vec::with_capacity(protocol.rounds); arms.len()];
        let mut paired = vec![Vec::with_capacity(protocol.rounds); arms.len()];
        let mut blocks = Vec::new();
        let mut round = 0;
        while round < protocol.rounds {
            let block_len = protocol.block_rounds.min(protocol.rounds - round);
            let load_start = load();
            for block_round in 0..block_len {
                let rotation = (round + block_round) % arms.len();
                let mut times = vec![0.0f64; arms.len()];
                for step in 0..arms.len() {
                    let index = (step + rotation) % arms.len();
                    let (arm, k) = arms[index];
                    times[index] = bench.call(arm, k)?;
                }
                for (index, &time) in times.iter().enumerate() {
                    samples[index].push(time);
                    paired[index].push(time / times[baseline]);
                }
            }
            let load_end = load();
            println!(
                "[verify-k] lane {} depth {} rounds {}..{}: load {:.2} -> {:.2}",
                bench.lane,
                bench.depth,
                round,
                round + block_len,
                load_start.loadavg_1_5_15[0],
                load_end.loadavg_1_5_15[0]
            );
            blocks.push(Block {
                first_round: round,
                rounds: block_len,
                load_start,
                load_end,
            });
            round += block_len;
        }
        let baseline_median = stats(&samples[baseline]).median_ms;
        let cells = arms
            .iter()
            .enumerate()
            .map(|(index, &(arm, k))| {
                let wall = stats(&samples[index]);
                Cell {
                    arm,
                    k,
                    ratio_of_medians: wall.median_ms / baseline_median,
                    wall,
                    paired_ratio: Some(stats(&paired[index])),
                }
            })
            .collect();
        Ok((blocks, cells))
    }

    /// Trace mode: for each K, one long uninterrupted run of batched verify
    /// calls, with two seconds of GPU idle before it and after the last one,
    /// so a Metal System Trace shows each (depth, K) phase as its own cluster
    /// of command buffers. Each call is exactly one command buffer. Phase
    /// boundaries are printed as seconds since process start.
    fn trace_phases(
        mut bench: Bench<'_>,
        protocol: &Protocol,
        started: Instant,
    ) -> Result<(Vec<Block>, Vec<Cell>)> {
        for _ in 0..protocol.warmup {
            for &k in &protocol.ks {
                bench.call(Arm::Batch, k)?;
            }
        }
        let mut blocks = Vec::new();
        let mut samples = Vec::new();
        for &k in &protocol.ks {
            std::thread::sleep(Duration::from_secs(2));
            let load_start = load();
            let phase_start = started.elapsed().as_secs_f64();
            let mut times = Vec::with_capacity(protocol.rounds);
            for _ in 0..protocol.rounds {
                times.push(bench.call(Arm::Batch, k)?);
            }
            let phase_end = started.elapsed().as_secs_f64();
            let load_end = load();
            println!(
                "[verify-k] TRACE PHASE lane={} depth={} K={k} calls={} \
                 t=+{phase_start:.3}s..+{phase_end:.3}s median={:.3}ms load {:.2}",
                bench.lane,
                bench.depth,
                protocol.rounds,
                stats(&times).median_ms,
                load_start.loadavg_1_5_15[0]
            );
            blocks.push(Block {
                first_round: 0,
                rounds: protocol.rounds,
                load_start,
                load_end,
            });
            samples.push((k, times));
        }
        // Idle tail so the last phase does not merge with the next depth's
        // prefill and warm-up in the trace.
        std::thread::sleep(Duration::from_secs(2));
        let baseline_median = samples
            .iter()
            .find(|(k, _)| *k == 1)
            .map(|(_, times)| stats(times).median_ms)
            .context("K=1 must be measured: it is the ratio denominator")?;
        let cells = samples
            .iter()
            .map(|(k, times)| {
                let wall = stats(times);
                Cell {
                    arm: Arm::Batch,
                    k: *k,
                    ratio_of_medians: wall.median_ms / baseline_median,
                    wall,
                    // Phases run back to back, not interleaved, so there is
                    // no same-round K=1 sample to pair with.
                    paired_ratio: None,
                }
            })
            .collect();
        Ok((blocks, cells))
    }

    fn stats(values: &[f64]) -> Stats {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        // Nearest rank: rank = ceil(q * n), 1-based.
        let rank = |q: f64| {
            if sorted.is_empty() {
                return f64::NAN;
            }
            let rank = (q * sorted.len() as f64).ceil() as usize;
            sorted[rank.clamp(1, sorted.len()) - 1]
        };
        Stats {
            samples: sorted.len(),
            median_ms: rank(0.5),
            p10_ms: rank(0.1),
            p90_ms: rank(0.9),
            min_ms: sorted.first().copied().unwrap_or(f64::NAN),
            max_ms: sorted.last().copied().unwrap_or(f64::NAN),
        }
    }

    fn batch_median(depth: &DepthReport, k: usize) -> Option<f64> {
        depth
            .cells
            .iter()
            .find(|cell| cell.arm == Arm::Batch && cell.k == k)
            .map(|cell| cell.wall.median_ms)
    }

    /// Splits each K's extra batched cost (over K=1) into the part that grows
    /// with cache depth and the part already present at the short depth.
    fn decompose(depths: &[DepthReport], ks: &[usize]) -> Vec<Decomposition> {
        let (Some(short), Some(long)) = (depths.first(), depths.last()) else {
            return Vec::new();
        };
        ks.iter()
            .filter(|&&k| k != 1)
            .filter_map(|&k| {
                let extra_ms_short = batch_median(short, k)? - batch_median(short, 1)?;
                let extra_ms_long = batch_median(long, k)? - batch_median(long, 1)?;
                let depth_dependent_extra_ms = extra_ms_long - extra_ms_short;
                Some(Decomposition {
                    k,
                    extra_ms_short,
                    extra_ms_long,
                    depth_dependent_extra_ms,
                    flat_extra_ms: extra_ms_short,
                    depth_dependent_fraction_at_long: depth_dependent_extra_ms / extra_ms_long,
                })
            })
            .collect()
    }

    fn print_summary(report: &Report) {
        println!();
        println!(
            "[verify-k] SUMMARY mode={} (wall ms per verify call; ratio = median(K)/median(batch K=1))",
            report.mode
        );
        for lane in &report.lanes {
            for depth in &lane.depths {
                println!("  lane {} depth {}", lane.lane, depth.depth);
                println!(
                    "    {:<10} {:>3} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}",
                    "arm", "K", "median", "p10", "p90", "min", "max", "ratio", "paired"
                );
                for cell in &depth.cells {
                    let arm = match cell.arm {
                        Arm::Batch => "batch",
                        Arm::Sequential => "sequential",
                    };
                    let paired = cell
                        .paired_ratio
                        .as_ref()
                        .map_or("-".to_string(), |paired| format!("{:.3}", paired.median_ms));
                    println!(
                        "    {:<10} {:>3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>8.3} {:>8}",
                        arm,
                        cell.k,
                        cell.wall.median_ms,
                        cell.wall.p10_ms,
                        cell.wall.p90_ms,
                        cell.wall.min_ms,
                        cell.wall.max_ms,
                        cell.ratio_of_medians,
                        paired
                    );
                }
            }
            for row in &lane.decomposition {
                println!(
                    "  lane {} K={} extra over K=1: short {:.3} ms, long {:.3} ms; \
                     depth-dependent {:.3} ms ({:.0}% of long-depth extra)",
                    lane.lane,
                    row.k,
                    row.extra_ms_short,
                    row.extra_ms_long,
                    row.depth_dependent_extra_ms,
                    row.depth_dependent_fraction_at_long * 100.0
                );
            }
        }
        println!();
    }
    fn machine() -> String {
        let chip = sysctl("machdep.cpu.brand_string");
        let memory = sysctl("hw.memsize")
            .parse::<u64>()
            .map(|bytes| format!("{} GiB", bytes >> 30))
            .unwrap_or_default();
        format!("{chip} {memory}").trim().to_string()
    }

    fn sysctl(key: &str) -> String {
        Command::new("sysctl")
            .args(["-n", key])
            .output()
            .ok()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_default()
    }

    fn load() -> Load {
        let mut loadavg_1_5_15 = [f64::NAN; 3];
        let raw = sysctl("vm.loadavg");
        for (slot, value) in loadavg_1_5_15.iter_mut().zip(
            raw.split_whitespace()
                .map(|token| token.trim_matches(|c| c == '{' || c == '}'))
                .filter_map(|token| token.parse::<f64>().ok()),
        ) {
            *slot = value;
        }
        let uptime = Command::new("uptime")
            .output()
            .ok()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_default();
        Load {
            loadavg_1_5_15,
            uptime,
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("verify_k_cost requires macOS with the Metal toolchain");
}
