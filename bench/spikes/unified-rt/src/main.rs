use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use safetensors::tensor::{Dtype as SafeDtype, SafeTensors};
use serde::Deserialize;
use synapse_bench::{
    parity::{load_corpus, load_reference, mean_parity, rank_overlap, Chunk},
    results::LaneResult,
};
use tokenizers::{Tokenizer, TruncationParams};

mod modernbert;
mod qwen3;

#[derive(Parser)]
#[command(name = "spike-unified-rt")]
struct Args {
    /// Path to a MiniLM or Qwen3 safetensors file or snapshot directory.
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Corpus JSONL ({id, path, text, tokens} per line).
    #[arg(long)]
    corpus: PathBuf,
    /// Optional cap for parity/throughput smoke runs.
    #[arg(long)]
    limit: Option<usize>,
    /// Output LaneResult JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Optional: write produced vectors (JSONL: {id, vec}). Alias kept for the spike prompt.
    #[arg(long = "vectors-out", alias = "emit-vectors")]
    vectors_out: Option<PathBuf>,
    /// Optional parity reference vectors (JSONL: {id, vec}).
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Minimum mean cosine when --reference is supplied.
    #[arg(long, default_value_t = 0.9999)]
    min_parity: f64,
    /// Minimum mean top-10 neighbor overlap when --reference is supplied.
    #[arg(long, default_value_t = 0.995)]
    min_rank_overlap: f64,
    /// Kernel provider to use.
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,
    /// Precision for the Metal resident encoder path.
    #[arg(long, value_enum, default_value_t = Precision::F32)]
    dtype: Precision,
    /// Metal graph execution strategy. Explicit O0 compilation is the serving default.
    #[arg(long, value_enum, default_value_t = Execution::Explicit)]
    execution: Execution,
    /// Optional directory for one compiled MPSGraph package per batch/sequence shape.
    #[arg(long)]
    package_cache: Option<PathBuf>,
    /// Tokenizer truncation max length.
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Greedy attention-unit budget per batch.
    #[arg(long, default_value_t = 4_000_000)]
    attention_units: usize,
    /// Optional model label for the result.
    #[arg(long)]
    model_label: Option<String>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum)]
enum DeviceArg {
    Cpu,
    Metal,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum)]
enum Precision {
    F32,
    F16,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum)]
enum Execution {
    Explicit,
    Lazy,
}

const GRAPH_REVISION: u32 = 2;

#[derive(Clone, Debug)]
struct MetalExecutionConfig {
    execution: Execution,
    package_root: Option<PathBuf>,
}

impl MetalExecutionConfig {
    fn from_args(args: &Args, family: &str) -> Result<Self> {
        let package_root = args.package_cache.as_ref().map(|root| {
            let model = fs::canonicalize(&args.model).unwrap_or_else(|_| args.model.clone());
            let identity = format!("{}", model.display());
            let hash = identity.bytes().fold(1469598103934665603u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
            });
            let os_build = std::process::Command::new("sw_vers")
                .arg("-buildVersion")
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
                .filter(|build| !build.is_empty())
                .unwrap_or_else(|| "unknown-os-build".to_owned());
            root.join(format!(
                "{family}-graph-v{GRAPH_REVISION}-{:016x}-{}-{os_build}",
                hash,
                args.dtype.as_str()
            ))
        });
        if let Some(root) = &package_root {
            fs::create_dir_all(root)
                .with_context(|| format!("create package cache {}", root.display()))?;
        }
        Ok(Self {
            execution: args.execution,
            package_root,
        })
    }

    fn package_path(&self, batch: usize, seq: usize) -> Option<PathBuf> {
        self.package_root
            .as_ref()
            .map(|root| root.join(format!("{batch}x{seq}.mpsgraphpackage")))
    }
}

impl Precision {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !(matches!(args.device, DeviceArg::Cpu) && matches!(args.dtype, Precision::F16)),
        "cpu + f16 is not supported for this spike; use --dtype f32 on cpu"
    );
    let started = Instant::now();

    let model = load_model_family(&args.model, args.dtype)?;
    let mut tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
    model.configure_tokenizer(&mut tokenizer, args.max_length)?;

    let execution = MetalExecutionConfig::from_args(&args, model.family_name())?;
    let mut provider = make_provider(args.device, args.dtype, execution)?;
    let _ = model.embed_batch(provider.as_mut(), &tokenizer, &["warmup"], args.max_length)?;
    let cold_load_s = started.elapsed().as_secs_f64();

    let chunks: Vec<Chunk> = load_corpus(&args.corpus, args.limit)?;
    let lengths = chunks
        .iter()
        .map(|chunk| model.token_length(&tokenizer, &chunk.text, args.max_length))
        .collect::<Result<Vec<_>>>()?;

    let mut order: Vec<usize> = (0..chunks.len()).collect();
    order.sort_by_key(|&index| lengths[index]);

    let infer_started = Instant::now();
    let mut input_tokens = 0u64;
    let mut items = 0u64;
    let mut produced_vectors: Vec<(String, Vec<f32>)> = Vec::with_capacity(chunks.len());

    let mut batch_start = 0usize;
    let mut batch_max_len = 0usize;
    let mut idx = 0usize;
    while idx <= order.len() {
        let flush = if idx == order.len() {
            idx > batch_start
        } else {
            let candidate_max = batch_max_len.max(lengths[order[idx]]);
            let count = idx - batch_start;
            count > 0 && (count + 1) * candidate_max * candidate_max > args.attention_units
        };
        if flush {
            let batch_indices = &order[batch_start..idx];
            let batch_texts: Vec<&str> = batch_indices
                .iter()
                .map(|&index| chunks[index].text.as_str())
                .collect();
            let vectors =
                model.embed_batch(provider.as_mut(), &tokenizer, &batch_texts, args.max_length)?;
            for (offset, vector) in vectors.into_iter().enumerate() {
                let original_index = batch_indices[offset];
                input_tokens += lengths[original_index] as u64;
                items += 1;
                produced_vectors.push((chunks[original_index].id.clone(), vector));
            }
            batch_start = idx;
            batch_max_len = 0;
            if idx == order.len() {
                break;
            }
        }
        if idx < order.len() {
            batch_max_len = batch_max_len.max(lengths[order[idx]]);
        }
        idx += 1;
    }
    let infer_wall_s = infer_started.elapsed().as_secs_f64();

    if let Some(path) = &args.vectors_out {
        write_vectors(path, &produced_vectors)?;
    }

    let parity_mean_cosine = match &args.reference {
        Some(path) => {
            let reference = load_reference(path)?;
            let (mean, matched) = mean_parity(
                produced_vectors
                    .iter()
                    .map(|(id, vector)| (id.clone(), vector.clone())),
                &reference,
            );
            let mean = mean.context("no overlapping ids with parity reference")?;
            model.validate_reference_coverage(matched, produced_vectors.len())?;
            let produced: HashMap<String, Vec<f32>> = produced_vectors.iter().cloned().collect();
            let ranks = rank_overlap(&produced, &reference, 10, 1)?;
            enforce_parity_gates(
                mean,
                ranks.mean_topk_overlap,
                args.min_parity,
                args.min_rank_overlap,
                matched,
                ranks.queries,
            )?;
            eprintln!(
                "parity gate: mean cosine {mean:.8}, top-10 rank overlap {:.6}",
                ranks.mean_topk_overlap
            );
            Some(mean)
        }
        None => None,
    };

    let result = LaneResult {
        lane: format!("owned-rt-{}", provider.name()),
        workload: "embed-corpus-v1".into(),
        model: args
            .model_label
            .unwrap_or_else(|| model.default_label(args.dtype)),
        cold_load_s,
        infer_wall_s,
        input_tokens,
        tok_per_s: input_tokens as f64 / infer_wall_s,
        items,
        parity_mean_cosine,
        self_peak_rss_bytes: None,
        notes: format!(
            "{}, provider={}, dtype={}, execution={:?}, package_cache={}, length-sorted attention_units={}, max_len={}; providers may override the model-family block, and Metal keeps block hidden states resident inside one MPSGraph per batch shape",
            model.notes(),
            provider.name(),
            args.dtype.as_str(),
            args.execution,
            args.package_cache.as_ref().map_or("disabled".into(), |path| path.display().to_string()),
            args.attention_units,
            args.max_length
        ),
    };

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, serde_json::to_string_pretty(&result)?)?;
    eprintln!(
        "{}: {} items, {} tokens, {:.1} tok/s, parity {:?}",
        result.lane, result.items, result.input_tokens, result.tok_per_s, result.parity_mean_cosine
    );
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn enforce_parity_gates(
    mean_cosine: f64,
    mean_top10_overlap: f64,
    min_cosine: f64,
    min_top10_overlap: f64,
    matched: usize,
    queries: usize,
) -> Result<()> {
    ensure!(
        mean_cosine >= min_cosine,
        "mean parity {mean_cosine:.8} below minimum {min_cosine:.8} over {matched} vectors"
    );
    ensure!(
        mean_top10_overlap >= min_top10_overlap,
        "mean top-10 rank overlap {mean_top10_overlap:.6} below minimum {min_top10_overlap:.6} over {queries} queries"
    );
    Ok(())
}

/// A loaded model family owns every policy that varies between embedding graphs.
///
/// Detection remains in the registry because it runs before a model exists. Keeping
/// loading, tokenizer behavior, token accounting, pooling, labels, and provider-hook
/// installation behind this object-safe seam lets the batch runner stay unchanged
/// when another family is registered.
trait ModelFamily {
    fn family_name(&self) -> &'static str;

    fn configure_tokenizer(&self, tokenizer: &mut Tokenizer, max_length: usize) -> Result<()> {
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|error| anyhow::anyhow!("truncation: {error}"))?;
        Ok(())
    }

    fn token_length(&self, tokenizer: &Tokenizer, text: &str, max_length: usize) -> Result<usize>;

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
        max_length: usize,
    ) -> Result<Vec<Vec<f32>>>;

    fn validate_reference_coverage(&self, _matched: usize, _produced: usize) -> Result<()> {
        Ok(())
    }

    fn default_label(&self, precision: Precision) -> String;
    fn notes(&self) -> String;
}

struct FamilyRegistration {
    detect: fn(&serde_json::Value) -> bool,
    load: fn(&Path, Precision) -> Result<Box<dyn ModelFamily>>,
}

fn load_model_family(path: &Path, precision: Precision) -> Result<Box<dyn ModelFamily>> {
    let root = resolve_model_root(path)?;
    let config_path = root.join("config.json");
    let config: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&config_path)
            .with_context(|| format!("read config {}", config_path.display()))?,
    )
    .with_context(|| format!("parse config {}", config_path.display()))?;
    let registry = [
        FamilyRegistration {
            detect: modernbert::detect_config,
            load: modernbert::load_family,
        },
        FamilyRegistration {
            detect: qwen3::detect_config,
            load: qwen3::load_family,
        },
        FamilyRegistration {
            detect: detect_minilm_config,
            load: |path, precision| Ok(Box::new(BertModel::load(path, precision)?)),
        },
    ];
    let registration = registry
        .iter()
        .find(|registration| (registration.detect)(&config))
        .context("config.json does not describe a supported embedding model family")?;
    (registration.load)(path, precision)
}

fn detect_minilm_config(config: &serde_json::Value) -> bool {
    config.get("model_type").and_then(serde_json::Value::as_str) == Some("bert")
}

fn make_provider(
    device: DeviceArg,
    dtype: Precision,
    execution: MetalExecutionConfig,
) -> Result<Box<dyn KernelProvider>> {
    match device {
        DeviceArg::Cpu => Ok(Box::new(CpuProvider)),
        DeviceArg::Metal => Ok(Box::new(MetalProvider::new_with_config(dtype, execution)?)),
    }
}

fn write_vectors(path: &Path, vectors: &[(String, Vec<f32>)]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(fs::File::create(path)?);
    for (id, vector) in vectors {
        serde_json::to_writer(&mut writer, &serde_json::json!({ "id": id, "vec": vector }))?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

type BlockContextFactory = fn(Precision, MetalExecutionConfig) -> Result<Box<dyn Any>>;

/// A block request keeps family-specific graph inputs inside the family while the
/// provider owns context lifetime and reuse. Because embedding graphs have different
/// typed parameters, erasing those parameters into a universal tensor schema would
/// invent unsupported generality; a family key plus a typed context callback gives all
/// providers one block-level dispatch surface without central family matches.
struct BlockForwardRequest<'a> {
    family: &'static str,
    create_context: BlockContextFactory,
    run: &'a mut dyn FnMut(&mut dyn Any) -> Result<()>,
}

#[allow(clippy::too_many_arguments)]
trait KernelProvider {
    fn name(&self) -> &'static str;

    fn matmul(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()>;

    fn matmul_static_rhs(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()> {
        self.matmul(m, n, k, a, b, b_layout, c)
    }

    fn block_forward(&mut self, _request: BlockForwardRequest<'_>) -> Result<bool> {
        Ok(false)
    }

    fn layer_norm(
        &mut self,
        rows: usize,
        hidden: usize,
        data: &mut [f32],
        weight: &[f32],
        bias: &[f32],
        eps: f32,
    ) -> Result<()> {
        ensure!(
            data.len() == rows * hidden,
            "layer_norm data shape mismatch"
        );
        ensure!(
            weight.len() == hidden && bias.len() == hidden,
            "layer_norm parameter shape mismatch"
        );
        for row in 0..rows {
            let start = row * hidden;
            let row_data = &mut data[start..start + hidden];
            let mean = row_data.iter().copied().sum::<f32>() / hidden as f32;
            let var = row_data
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / hidden as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..hidden {
                row_data[i] = (row_data[i] - mean) * inv * weight[i] + bias[i];
            }
        }
        Ok(())
    }
}

#[derive(Copy, Clone)]
enum BLayout {
    RowMajorKn,
    RowMajorNkTransposed,
}

struct CpuProvider;

impl KernelProvider for CpuProvider {
    fn name(&self) -> &'static str {
        "cpu-accelerate"
    }

    fn matmul(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()> {
        ensure!(a.len() == m * k, "matmul A shape mismatch");
        ensure!(b.len() == n * k, "matmul B shape mismatch");
        ensure!(c.len() == m * n, "matmul C shape mismatch");
        matmul_impl(m, n, k, a, b, b_layout, c);
        Ok(())
    }
}

struct MetalProvider {
    context: metal_backend::MpsGraphContext,
    block_contexts: HashMap<&'static str, Box<dyn Any>>,
    dtype: Precision,
    execution: MetalExecutionConfig,
}

impl MetalProvider {
    #[cfg(test)]
    fn new(dtype: Precision) -> Result<Self> {
        Self::new_with_config(
            dtype,
            MetalExecutionConfig {
                execution: Execution::Lazy,
                package_root: None,
            },
        )
    }

    fn new_with_config(dtype: Precision, execution: MetalExecutionConfig) -> Result<Self> {
        Ok(Self {
            context: metal_backend::MpsGraphContext::new_with_config(execution.clone())?,
            block_contexts: HashMap::new(),
            dtype,
            execution,
        })
    }
}

impl KernelProvider for MetalProvider {
    fn name(&self) -> &'static str {
        match self.dtype {
            Precision::F32 => "metal-mpsgraph-resident-encoder-fp32",
            Precision::F16 => "metal-mpsgraph-resident-encoder-f16",
        }
    }

    fn matmul(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()> {
        ensure!(a.len() == m * k, "matmul A shape mismatch");
        ensure!(b.len() == n * k, "matmul B shape mismatch");
        ensure!(c.len() == m * n, "matmul C shape mismatch");
        self.context
            .matmul(m, n, k, a, b, b_layout, c, false, self.dtype)
    }

    fn matmul_static_rhs(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()> {
        ensure!(a.len() == m * k, "matmul A shape mismatch");
        ensure!(b.len() == n * k, "matmul B shape mismatch");
        ensure!(c.len() == m * n, "matmul C shape mismatch");
        self.context
            .matmul(m, n, k, a, b, b_layout, c, true, self.dtype)
    }

    fn block_forward(&mut self, request: BlockForwardRequest<'_>) -> Result<bool> {
        if !self.block_contexts.contains_key(request.family) {
            let context = (request.create_context)(self.dtype, self.execution.clone())?;
            self.block_contexts.insert(request.family, context);
        }
        let context = self
            .block_contexts
            .get_mut(request.family)
            .expect("block context inserted above");
        (request.run)(context.as_mut())?;
        Ok(true)
    }
}

#[cfg(target_os = "macos")]
mod metal_backend {
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::ptr::NonNull;

    use anyhow::{bail, ensure, Result};

    use super::{
        decode_f16_bits, encode_f16_bits, BLayout, EncoderLayer, Execution, MetalExecutionConfig,
        Precision,
    };

    #[repr(C)]
    struct SynapseMpsEncoderLayerParams {
        query_weight: *const c_void,
        query_bias: *const c_void,
        key_weight: *const c_void,
        key_bias: *const c_void,
        value_weight: *const c_void,
        value_bias: *const c_void,
        attention_output_weight: *const c_void,
        attention_output_bias: *const c_void,
        attention_ln_weight: *const c_void,
        attention_ln_bias: *const c_void,
        intermediate_weight: *const c_void,
        intermediate_bias: *const c_void,
        output_weight: *const c_void,
        output_bias: *const c_void,
        output_ln_weight: *const c_void,
        output_ln_bias: *const c_void,
    }

    #[repr(i32)]
    #[derive(Copy, Clone)]
    enum SynapseMpsDType {
        Float32 = 0,
        Float16 = 1,
    }

    impl From<Precision> for SynapseMpsDType {
        fn from(value: Precision) -> Self {
            match value {
                Precision::F32 => Self::Float32,
                Precision::F16 => Self::Float16,
            }
        }
    }

    pub struct MpsGraphContext {
        raw: NonNull<c_void>,
        execution: MetalExecutionConfig,
    }

    impl MpsGraphContext {
        pub fn new_with_config(execution: MetalExecutionConfig) -> Result<Self> {
            let raw = unsafe { synapse_mps_context_new() };
            let raw = NonNull::new(raw).ok_or_else(last_error)?;
            Ok(Self { raw, execution })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn matmul(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            a: &[f32],
            b: &[f32],
            b_layout: BLayout,
            c: &mut [f32],
            cache_rhs: bool,
            dtype: Precision,
        ) -> Result<()> {
            let b_is_row_major_nk = match b_layout {
                BLayout::RowMajorKn => 0,
                BLayout::RowMajorNkTransposed => 1,
            };
            let ffi_dtype = SynapseMpsDType::from(dtype) as i32;
            let status = match dtype {
                Precision::F32 => unsafe {
                    synapse_mps_matmul(
                        self.raw.as_ptr(),
                        m as u64,
                        n as u64,
                        k as u64,
                        a.as_ptr().cast(),
                        b.as_ptr().cast(),
                        ffi_dtype,
                        b_is_row_major_nk,
                        c.as_mut_ptr().cast(),
                        i32::from(cache_rhs),
                    )
                },
                Precision::F16 => {
                    let a_half = encode_f16_bits(a);
                    let b_half = encode_f16_bits(b);
                    let mut output_half = vec![0u16; c.len()];
                    let status = unsafe {
                        synapse_mps_matmul(
                            self.raw.as_ptr(),
                            m as u64,
                            n as u64,
                            k as u64,
                            a_half.as_ptr().cast(),
                            b_half.as_ptr().cast(),
                            ffi_dtype,
                            b_is_row_major_nk,
                            output_half.as_mut_ptr().cast(),
                            i32::from(cache_rhs),
                        )
                    };
                    if status == 0 {
                        c.copy_from_slice(&decode_f16_bits(&output_half));
                    }
                    status
                }
            };
            if status != 0 {
                bail!(
                    "MPSGraph matmul failed with status {status}: {}",
                    last_error()
                );
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encoder_forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            layer_norm_eps: f32,
            layers: &[EncoderLayer],
            dtype: Precision,
        ) -> Result<()> {
            ensure!(
                batch > 0 && seq > 0 && hidden > 0 && heads > 0 && intermediate > 0,
                "encoder dimensions must be non-zero"
            );
            ensure!(hidden % heads == 0, "hidden size must divide heads");
            ensure!(
                hidden_states.len() == batch * seq * hidden,
                "encoder hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == batch * seq,
                "encoder mask shape mismatch"
            );
            ensure!(!layers.is_empty(), "encoder requires at least one layer");

            let hidden_hidden = hidden * hidden;
            let intermediate_hidden = intermediate * hidden;
            let hidden_intermediate = hidden * intermediate;
            for (index, layer) in layers.iter().enumerate() {
                ensure!(
                    layer.query.weight.data.len() == hidden_hidden
                        && layer.key.weight.data.len() == hidden_hidden
                        && layer.value.weight.data.len() == hidden_hidden
                        && layer.attention_output.weight.data.len() == hidden_hidden,
                    "encoder layer {index} attention weight shape mismatch"
                );
                ensure!(
                    layer.query.bias.len() == hidden
                        && layer.key.bias.len() == hidden
                        && layer.value.bias.len() == hidden
                        && layer.attention_output.bias.len() == hidden
                        && layer.attention_ln_weight.len() == hidden
                        && layer.attention_ln_bias.len() == hidden
                        && layer.output_ln_weight.len() == hidden
                        && layer.output_ln_bias.len() == hidden,
                    "encoder layer {index} hidden vector shape mismatch"
                );
                ensure!(
                    layer.intermediate.weight.data.len() == intermediate_hidden
                        && layer.intermediate.bias.len() == intermediate,
                    "encoder layer {index} intermediate shape mismatch"
                );
                ensure!(
                    layer.output.weight.data.len() == hidden_intermediate
                        && layer.output.bias.len() == hidden,
                    "encoder layer {index} output shape mismatch"
                );
            }

            let additive_mask: Vec<f32> = attention_mask
                .iter()
                .map(|&mask| if mask == 0 { -10_000.0 } else { 0.0 })
                .collect();
            let params: Vec<SynapseMpsEncoderLayerParams> = match dtype {
                Precision::F32 => layers
                    .iter()
                    .map(|layer| SynapseMpsEncoderLayerParams {
                        query_weight: layer.query.weight.data.as_ptr().cast(),
                        query_bias: layer.query.bias.as_slice().as_ptr().cast(),
                        key_weight: layer.key.weight.data.as_ptr().cast(),
                        key_bias: layer.key.bias.as_slice().as_ptr().cast(),
                        value_weight: layer.value.weight.data.as_ptr().cast(),
                        value_bias: layer.value.bias.as_slice().as_ptr().cast(),
                        attention_output_weight: layer.attention_output.weight.data.as_ptr().cast(),
                        attention_output_bias: layer
                            .attention_output
                            .bias
                            .as_slice()
                            .as_ptr()
                            .cast(),
                        attention_ln_weight: layer.attention_ln_weight.as_slice().as_ptr().cast(),
                        attention_ln_bias: layer.attention_ln_bias.as_slice().as_ptr().cast(),
                        intermediate_weight: layer.intermediate.weight.data.as_ptr().cast(),
                        intermediate_bias: layer.intermediate.bias.as_slice().as_ptr().cast(),
                        output_weight: layer.output.weight.data.as_ptr().cast(),
                        output_bias: layer.output.bias.as_slice().as_ptr().cast(),
                        output_ln_weight: layer.output_ln_weight.as_slice().as_ptr().cast(),
                        output_ln_bias: layer.output_ln_bias.as_slice().as_ptr().cast(),
                    })
                    .collect(),
                Precision::F16 => layers
                    .iter()
                    .map(|layer| {
                        Ok(SynapseMpsEncoderLayerParams {
                            query_weight: layer.query.weight.metal_f16_bits()?.as_ptr().cast(),
                            query_bias: layer.query.bias.metal_f16_bits()?.as_ptr().cast(),
                            key_weight: layer.key.weight.metal_f16_bits()?.as_ptr().cast(),
                            key_bias: layer.key.bias.metal_f16_bits()?.as_ptr().cast(),
                            value_weight: layer.value.weight.metal_f16_bits()?.as_ptr().cast(),
                            value_bias: layer.value.bias.metal_f16_bits()?.as_ptr().cast(),
                            attention_output_weight: layer
                                .attention_output
                                .weight
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            attention_output_bias: layer
                                .attention_output
                                .bias
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            attention_ln_weight: layer
                                .attention_ln_weight
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            attention_ln_bias: layer
                                .attention_ln_bias
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            intermediate_weight: layer
                                .intermediate
                                .weight
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            intermediate_bias: layer
                                .intermediate
                                .bias
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            output_weight: layer.output.weight.metal_f16_bits()?.as_ptr().cast(),
                            output_bias: layer.output.bias.metal_f16_bits()?.as_ptr().cast(),
                            output_ln_weight: layer
                                .output_ln_weight
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            output_ln_bias: layer.output_ln_bias.metal_f16_bits()?.as_ptr().cast(),
                        })
                    })
                    .collect::<Result<_>>()?,
            };
            let ffi_dtype = SynapseMpsDType::from(dtype) as i32;
            if matches!(self.execution.execution, Execution::Explicit) {
                let package = self.execution.package_path(batch, seq);
                let load_package = package.as_ref().is_some_and(|path| path.exists());
                let package_c = package
                    .as_ref()
                    .map(|path| CString::new(path.to_string_lossy().as_bytes()))
                    .transpose()?;
                let mut prepare_wall_s = 0.0;
                let mut specialize_wall_s = 0.0;
                let mut serialize_wall_s = 0.0;
                let prepare_status = unsafe {
                    synapse_mps_prepare_encoder(
                        self.raw.as_ptr(),
                        batch as u64,
                        seq as u64,
                        hidden as u64,
                        heads as u64,
                        intermediate as u64,
                        layers.len() as u64,
                        layer_norm_eps,
                        ffi_dtype,
                        0,
                        package_c
                            .as_ref()
                            .map_or(std::ptr::null(), |path| path.as_ptr()),
                        i32::from(load_package),
                        0,
                        &mut prepare_wall_s,
                        &mut specialize_wall_s,
                        &mut serialize_wall_s,
                    )
                };
                if prepare_status != 0 {
                    bail!(
                        "MPSGraph encoder preparation failed with status {prepare_status}: {}",
                        last_error()
                    );
                }
                eprintln!(
                    "Metal executable {} {}x{}: prepare={prepare_wall_s:.6}s specialize={specialize_wall_s:.6}s serialize={serialize_wall_s:.6}s",
                    if load_package { "loaded" } else { "compiled" }, batch, seq
                );
            }
            let status = match dtype {
                Precision::F32 => {
                    let mut output = vec![0.0f32; hidden_states.len()];
                    let status = unsafe {
                        synapse_mps_encoder_forward(
                            self.raw.as_ptr(),
                            batch as u64,
                            seq as u64,
                            hidden as u64,
                            heads as u64,
                            intermediate as u64,
                            layers.len() as u64,
                            layer_norm_eps,
                            ffi_dtype,
                            hidden_states.as_ptr().cast(),
                            additive_mask.as_ptr(),
                            output.as_mut_ptr().cast(),
                            params.as_ptr(),
                        )
                    };
                    if status == 0 {
                        hidden_states.copy_from_slice(&output);
                    }
                    status
                }
                Precision::F16 => {
                    let input_half = encode_f16_bits(hidden_states);
                    let mut output_half = vec![0u16; hidden_states.len()];
                    let status = unsafe {
                        synapse_mps_encoder_forward(
                            self.raw.as_ptr(),
                            batch as u64,
                            seq as u64,
                            hidden as u64,
                            heads as u64,
                            intermediate as u64,
                            layers.len() as u64,
                            layer_norm_eps,
                            ffi_dtype,
                            input_half.as_ptr().cast(),
                            additive_mask.as_ptr(),
                            output_half.as_mut_ptr().cast(),
                            params.as_ptr(),
                        )
                    };
                    if status == 0 {
                        hidden_states.copy_from_slice(&decode_f16_bits(&output_half));
                    }
                    status
                }
            };
            if status != 0 {
                bail!(
                    "MPSGraph encoder forward failed with status {status}: {}",
                    last_error()
                );
            }
            Ok(())
        }
    }

    impl Drop for MpsGraphContext {
        fn drop(&mut self) {
            unsafe { synapse_mps_context_free(self.raw.as_ptr()) }
        }
    }

    fn last_error() -> anyhow::Error {
        unsafe {
            let raw = synapse_mps_last_error();
            if raw.is_null() {
                return anyhow::anyhow!("unknown MPSGraph error");
            }
            let message = CStr::from_ptr(raw).to_string_lossy();
            if message.is_empty() {
                anyhow::anyhow!("unknown MPSGraph error")
            } else {
                anyhow::anyhow!(message.into_owned())
            }
        }
    }

    unsafe extern "C" {
        fn synapse_mps_context_new() -> *mut c_void;
        fn synapse_mps_context_free(context: *mut c_void);
        fn synapse_mps_matmul(
            context: *mut c_void,
            m: u64,
            n: u64,
            k: u64,
            a: *const c_void,
            b: *const c_void,
            dtype: i32,
            b_is_row_major_nk: i32,
            c: *mut c_void,
            cache_rhs: i32,
        ) -> i32;
        fn synapse_mps_prepare_encoder(
            context: *mut c_void,
            batch: u64,
            seq: u64,
            hidden: u64,
            heads: u64,
            intermediate: u64,
            layer_count: u64,
            layer_norm_eps: f32,
            dtype: i32,
            optimization_level: i32,
            package_path: *const c_char,
            load_package: i32,
            append_package: i32,
            prepare_wall_s: *mut f64,
            specialize_wall_s: *mut f64,
            serialize_wall_s: *mut f64,
        ) -> i32;
        fn synapse_mps_encoder_forward(
            context: *mut c_void,
            batch: u64,
            seq: u64,
            hidden: u64,
            heads: u64,
            intermediate: u64,
            layer_count: u64,
            layer_norm_eps: f32,
            dtype: i32,
            input: *const c_void,
            additive_mask: *const f32,
            output: *mut c_void,
            layers: *const SynapseMpsEncoderLayerParams,
        ) -> i32;
        fn synapse_mps_last_error() -> *const c_char;
    }
}

#[cfg(not(target_os = "macos"))]
mod metal_backend {
    use anyhow::{bail, Result};

    use super::{BLayout, EncoderLayer, Precision};

    pub struct MpsGraphContext;

    impl MpsGraphContext {
        pub fn new_with_config(_execution: super::MetalExecutionConfig) -> Result<Self> {
            bail!("Metal MPSGraph provider is only available on macOS")
        }

        pub fn matmul(
            &mut self,
            _m: usize,
            _n: usize,
            _k: usize,
            _a: &[f32],
            _b: &[f32],
            _b_layout: BLayout,
            _c: &mut [f32],
            _cache_rhs: bool,
            _dtype: Precision,
        ) -> Result<()> {
            bail!("Metal MPSGraph provider is only available on macOS")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encoder_forward(
            &mut self,
            _hidden_states: &mut [f32],
            _attention_mask: &[u8],
            _batch: usize,
            _seq: usize,
            _hidden: usize,
            _heads: usize,
            _intermediate: usize,
            _layer_norm_eps: f32,
            _layers: &[EncoderLayer],
            _dtype: Precision,
        ) -> Result<()> {
            bail!("Metal MPSGraph provider is only available on macOS")
        }
    }
}

#[cfg(target_os = "macos")]
fn matmul_impl(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    b_layout: BLayout,
    c: &mut [f32],
) {
    use std::os::raw::c_int;

    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }

    let trans_b = match b_layout {
        BLayout::RowMajorKn => CBLAS_NO_TRANS,
        BLayout::RowMajorNkTransposed => CBLAS_TRANS,
    };
    let ldb = match b_layout {
        BLayout::RowMajorKn => n,
        BLayout::RowMajorNkTransposed => k,
    } as c_int;

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            trans_b,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a.as_ptr(),
            k as c_int,
            b.as_ptr(),
            ldb,
            0.0,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn matmul_impl(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    b_layout: BLayout,
    c: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let b_value = match b_layout {
                    BLayout::RowMajorKn => b[p * n + j],
                    BLayout::RowMajorNkTransposed => b[j * k + p],
                };
                sum += a[i * k + p] * b_value;
            }
            c[i * n + j] = sum;
        }
    }
}

#[derive(Clone, Debug)]
struct Tensor {
    dtype: TensorDType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
    metal_f16_bits: Option<Vec<u16>>,
}

#[derive(Copy, Clone, Debug)]
enum TensorDType {
    F32,
}

#[derive(Clone, Debug)]
struct ParamVector {
    values: Vec<f32>,
    metal_f16_bits: Option<Vec<u16>>,
}

impl ParamVector {
    fn new(values: Vec<f32>) -> Self {
        Self {
            values,
            metal_f16_bits: None,
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn as_slice(&self) -> &[f32] {
        &self.values
    }

    fn prepare_metal_f16(&mut self) {
        if self.metal_f16_bits.is_none() {
            self.metal_f16_bits = Some(encode_f16_bits(&self.values));
        }
    }

    fn metal_f16_bits(&self) -> Result<&[u16]> {
        self.metal_f16_bits
            .as_deref()
            .context("f16 mirror missing for Metal parameter")
    }
}

impl Tensor {
    fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected = shape.iter().product::<usize>();
        ensure!(
            expected == data.len(),
            "tensor data length {} does not match shape {:?}",
            data.len(),
            shape
        );
        Ok(Self {
            dtype: TensorDType::F32,
            strides: row_major_strides(&shape),
            shape,
            data,
            metal_f16_bits: None,
        })
    }

    fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>();
        Self {
            dtype: TensorDType::F32,
            strides: row_major_strides(&shape),
            shape,
            data: vec![0.0; len],
            metal_f16_bits: None,
        }
    }

    fn dim(&self, index: usize) -> usize {
        self.shape[index]
    }

    fn prepare_metal_f16(&mut self) {
        if self.metal_f16_bits.is_none() {
            self.metal_f16_bits = Some(encode_f16_bits(&self.data));
        }
    }

    fn metal_f16_bits(&self) -> Result<&[u16]> {
        self.metal_f16_bits
            .as_deref()
            .context("f16 mirror missing for Metal tensor")
    }

    fn as_vector(&self) -> Result<&[f32]> {
        self.ensure_f32_contiguous()?;
        ensure!(
            self.shape.len() == 1,
            "expected vector tensor, got {:?}",
            self.shape
        );
        Ok(&self.data)
    }

    fn matrix_shape(&self) -> Result<(usize, usize)> {
        self.ensure_f32_contiguous()?;
        ensure!(
            self.shape.len() == 2,
            "expected matrix tensor, got {:?}",
            self.shape
        );
        Ok((self.shape[0], self.shape[1]))
    }

    fn ensure_f32_contiguous(&self) -> Result<()> {
        ensure!(
            matches!(self.dtype, TensorDType::F32),
            "only f32 tensors are executable"
        );
        ensure!(
            self.strides == row_major_strides(&self.shape),
            "only contiguous row-major tensors are executable"
        );
        Ok(())
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

fn encode_f16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect()
}

fn decode_f16_bits(values: &[u16]) -> Vec<f32> {
    values
        .iter()
        .map(|&value| half::f16::from_bits(value).to_f32())
        .collect()
}

#[derive(Deserialize)]
struct BertConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
}

fn default_type_vocab_size() -> usize {
    2
}

fn default_layer_norm_eps() -> f32 {
    1e-12
}

fn default_hidden_act() -> String {
    "gelu".to_string()
}

struct BertModel {
    config: BertConfig,
    embeddings: Embeddings,
    layers: Vec<EncoderLayer>,
}

struct Embeddings {
    word: Tensor,
    position: Tensor,
    token_type: Tensor,
    layer_norm_weight: ParamVector,
    layer_norm_bias: ParamVector,
}

struct EncoderLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    attention_ln_weight: ParamVector,
    attention_ln_bias: ParamVector,
    intermediate: Linear,
    output: Linear,
    output_ln_weight: ParamVector,
    output_ln_bias: ParamVector,
}

struct Linear {
    weight: Tensor,
    bias: ParamVector,
}

impl EncoderLayer {
    fn prepare_metal_f16(&mut self) {
        self.query.prepare_metal_f16();
        self.key.prepare_metal_f16();
        self.value.prepare_metal_f16();
        self.attention_output.prepare_metal_f16();
        self.attention_ln_weight.prepare_metal_f16();
        self.attention_ln_bias.prepare_metal_f16();
        self.intermediate.prepare_metal_f16();
        self.output.prepare_metal_f16();
        self.output_ln_weight.prepare_metal_f16();
        self.output_ln_bias.prepare_metal_f16();
    }
}

impl Linear {
    fn prepare_metal_f16(&mut self) {
        self.weight.prepare_metal_f16();
        self.bias.prepare_metal_f16();
    }
}

impl BertModel {
    fn load(path: &Path, precision: Precision) -> Result<Self> {
        let model_root = resolve_model_root(path)?;
        let config_path = model_root.join("config.json");
        let config: BertConfig = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .with_context(|| format!("read config {}", config_path.display()))?,
        )
        .with_context(|| format!("parse config {}", config_path.display()))?;
        ensure!(
            config.hidden_act == "gelu" || config.hidden_act == "gelu_new",
            "unsupported hidden_act {}",
            config.hidden_act
        );
        ensure!(
            config.hidden_size % config.num_attention_heads == 0,
            "hidden size must divide heads"
        );

        let tensors = load_safetensor_map(&model_root, path)?;
        let embeddings = Embeddings {
            word: get_tensor(&tensors, "embeddings.word_embeddings.weight")?,
            position: get_tensor(&tensors, "embeddings.position_embeddings.weight")?,
            token_type: get_tensor(&tensors, "embeddings.token_type_embeddings.weight")?,
            layer_norm_weight: ParamVector::new(
                get_tensor(&tensors, "embeddings.LayerNorm.weight")?
                    .as_vector()?
                    .to_vec(),
            ),
            layer_norm_bias: ParamVector::new(
                get_tensor(&tensors, "embeddings.LayerNorm.bias")?
                    .as_vector()?
                    .to_vec(),
            ),
        };
        ensure!(
            embeddings.word.shape == vec![config.vocab_size, config.hidden_size],
            "word embedding shape {:?} does not match config",
            embeddings.word.shape
        );
        ensure!(
            embeddings.position.dim(0) >= config.max_position_embeddings.min(512),
            "position embedding table unexpectedly short"
        );
        ensure!(
            embeddings.token_type.dim(0) >= config.type_vocab_size.min(1),
            "token type embedding table unexpectedly short"
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{layer_index}");
            layers.push(EncoderLayer {
                query: Linear::load(&tensors, &format!("{prefix}.attention.self.query"))?,
                key: Linear::load(&tensors, &format!("{prefix}.attention.self.key"))?,
                value: Linear::load(&tensors, &format!("{prefix}.attention.self.value"))?,
                attention_output: Linear::load(
                    &tensors,
                    &format!("{prefix}.attention.output.dense"),
                )?,
                attention_ln_weight: ParamVector::new(
                    get_tensor(
                        &tensors,
                        &format!("{prefix}.attention.output.LayerNorm.weight"),
                    )?
                    .as_vector()?
                    .to_vec(),
                ),
                attention_ln_bias: ParamVector::new(
                    get_tensor(
                        &tensors,
                        &format!("{prefix}.attention.output.LayerNorm.bias"),
                    )?
                    .as_vector()?
                    .to_vec(),
                ),
                intermediate: Linear::load(&tensors, &format!("{prefix}.intermediate.dense"))?,
                output: Linear::load(&tensors, &format!("{prefix}.output.dense"))?,
                output_ln_weight: ParamVector::new(
                    get_tensor(&tensors, &format!("{prefix}.output.LayerNorm.weight"))?
                        .as_vector()?
                        .to_vec(),
                ),
                output_ln_bias: ParamVector::new(
                    get_tensor(&tensors, &format!("{prefix}.output.LayerNorm.bias"))?
                        .as_vector()?
                        .to_vec(),
                ),
            });
        }

        if matches!(precision, Precision::F16) {
            for layer in &mut layers {
                layer.prepare_metal_f16();
            }
        }

        Ok(Self {
            config,
            embeddings,
            layers,
        })
    }

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>> {
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| anyhow::anyhow!("encode_batch: {error}"))?;
        let batch = encodings.len();
        let seq = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .max()
            .unwrap_or(1)
            .max(1);
        let mut input_ids = vec![0u32; batch * seq];
        let mut attention_mask = vec![0u8; batch * seq];
        for (row, encoding) in encodings.iter().enumerate() {
            for (col, (&id, &mask)) in encoding
                .get_ids()
                .iter()
                .zip(encoding.get_attention_mask())
                .enumerate()
            {
                input_ids[row * seq + col] = id;
                attention_mask[row * seq + col] = mask as u8;
            }
        }

        let hidden = self.forward(provider, &input_ids, &attention_mask, batch, seq)?;
        Ok(mean_pool_l2(
            &hidden,
            &attention_mask,
            batch,
            seq,
            self.config.hidden_size,
        ))
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        input_ids: &[u32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
    ) -> Result<Tensor> {
        let hidden = self.config.hidden_size;
        let heads = self.config.num_attention_heads;
        let rows = batch * seq;
        let mut x = Tensor::zeros(vec![batch, seq, hidden]);

        for b in 0..batch {
            for s in 0..seq {
                let token_id = input_ids[b * seq + s] as usize;
                ensure!(
                    token_id < self.embeddings.word.dim(0),
                    "token id {token_id} outside vocab"
                );
                ensure!(
                    s < self.embeddings.position.dim(0),
                    "position {s} outside position embeddings"
                );
                let out = (b * seq + s) * hidden;
                for h in 0..hidden {
                    x.data[out + h] = self.embeddings.word.data[token_id * hidden + h]
                        + self.embeddings.position.data[s * hidden + h]
                        + self.embeddings.token_type.data[h];
                }
            }
        }
        provider.layer_norm(
            rows,
            hidden,
            &mut x.data,
            self.embeddings.layer_norm_weight.as_slice(),
            self.embeddings.layer_norm_bias.as_slice(),
            self.config.layer_norm_eps,
        )?;

        let mut run = |context: &mut dyn Any| {
            let context = context
                .downcast_mut::<MiniLmBlockContext>()
                .context("MiniLM provider returned the wrong block context type")?;
            context.graph.encoder_forward(
                &mut x.data,
                attention_mask,
                batch,
                seq,
                hidden,
                heads,
                self.config.intermediate_size,
                self.config.layer_norm_eps,
                &self.layers,
                context.precision,
            )
        };
        if provider.block_forward(BlockForwardRequest {
            family: self.family_name(),
            create_context: new_minilm_block_context,
            run: &mut run,
        })? {
            return Ok(x);
        }

        encoder_layers_scalar_forward(
            provider,
            &mut x.data,
            attention_mask,
            batch,
            seq,
            hidden,
            heads,
            self.config.intermediate_size,
            self.config.layer_norm_eps,
            &self.layers,
        )?;
        Ok(x)
    }
}

impl ModelFamily for BertModel {
    fn family_name(&self) -> &'static str {
        "minilm"
    }

    fn token_length(&self, tokenizer: &Tokenizer, text: &str, _max_length: usize) -> Result<usize> {
        tokenizer
            .encode(text, true)
            .map(|encoding| encoding.get_ids().len())
            .map_err(|error| anyhow::anyhow!("encode: {error}"))
    }

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
        _max_length: usize,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(provider, tokenizer, texts)
    }

    fn default_label(&self, precision: Precision) -> String {
        format!("all-MiniLM-L6-v2@owned-rt-{}", precision.as_str())
    }

    fn notes(&self) -> String {
        "direct BERT encoder, mean pool+l2 on CPU".to_owned()
    }
}

fn new_minilm_block_context(
    precision: Precision,
    execution: MetalExecutionConfig,
) -> Result<Box<dyn Any>> {
    Ok(Box::new(MiniLmBlockContext {
        graph: metal_backend::MpsGraphContext::new_with_config(execution)?,
        precision,
    }))
}

struct MiniLmBlockContext {
    graph: metal_backend::MpsGraphContext,
    precision: Precision,
}

#[allow(clippy::too_many_arguments)]
fn encoder_layers_scalar_forward(
    provider: &mut dyn KernelProvider,
    hidden_states: &mut [f32],
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    hidden: usize,
    heads: usize,
    intermediate_size: usize,
    layer_norm_eps: f32,
    layers: &[EncoderLayer],
) -> Result<()> {
    let rows = batch * seq;
    let head_dim = hidden / heads;
    let mut current = hidden_states.to_vec();
    for layer in layers {
        let residual = current.clone();
        let q = layer.query.forward(provider, rows, hidden, &current)?;
        let k = layer.key.forward(provider, rows, hidden, &current)?;
        let v = layer.value.forward(provider, rows, hidden, &current)?;
        let context = self_attention(
            provider,
            &q,
            &k,
            &v,
            attention_mask,
            batch,
            seq,
            heads,
            head_dim,
        )?;
        let mut attention_out = layer
            .attention_output
            .forward(provider, rows, hidden, &context)?;
        for (value, residual_value) in attention_out.iter_mut().zip(residual) {
            *value += residual_value;
        }
        provider.layer_norm(
            rows,
            hidden,
            &mut attention_out,
            layer.attention_ln_weight.as_slice(),
            layer.attention_ln_bias.as_slice(),
            layer_norm_eps,
        )?;

        let residual = attention_out.clone();
        let mut intermediate =
            layer
                .intermediate
                .forward(provider, rows, hidden, &attention_out)?;
        for value in &mut intermediate {
            *value = gelu(*value);
        }
        let mut output = layer
            .output
            .forward(provider, rows, intermediate_size, &intermediate)?;
        for (value, residual_value) in output.iter_mut().zip(residual) {
            *value += residual_value;
        }
        provider.layer_norm(
            rows,
            hidden,
            &mut output,
            layer.output_ln_weight.as_slice(),
            layer.output_ln_bias.as_slice(),
            layer_norm_eps,
        )?;
        current = output;
    }
    hidden_states.copy_from_slice(&current);
    Ok(())
}

impl Linear {
    fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let weight = get_tensor(tensors, &format!("{prefix}.weight"))?;
        let bias = ParamVector::new(
            get_tensor(tensors, &format!("{prefix}.bias"))?
                .as_vector()?
                .to_vec(),
        );
        Ok(Self { weight, bias })
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        rows: usize,
        input: usize,
        values: &[f32],
    ) -> Result<Vec<f32>> {
        let (output, weight_input) = self.weight.matrix_shape()?;
        ensure!(
            weight_input == input,
            "linear input mismatch: weight expects {weight_input}, got {input}"
        );
        ensure!(self.bias.len() == output, "linear bias shape mismatch");
        let bias = self.bias.as_slice();
        ensure!(values.len() == rows * input, "linear values shape mismatch");
        let mut out = vec![0.0f32; rows * output];
        provider.matmul_static_rhs(
            rows,
            output,
            input,
            values,
            &self.weight.data,
            BLayout::RowMajorNkTransposed,
            &mut out,
        )?;
        for row in 0..rows {
            let start = row * output;
            for col in 0..output {
                out[start + col] += bias[col];
            }
        }
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
fn self_attention(
    provider: &mut dyn KernelProvider,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let hidden = heads * head_dim;
    let mut context = vec![0.0f32; batch * seq * hidden];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut q_head = vec![0.0f32; seq * head_dim];
    let mut k_head = vec![0.0f32; seq * head_dim];
    let mut v_head = vec![0.0f32; seq * head_dim];
    let mut scores = vec![0.0f32; seq * seq];
    let mut ctx_head = vec![0.0f32; seq * head_dim];

    for b in 0..batch {
        for head in 0..heads {
            for s in 0..seq {
                let source = (b * seq + s) * hidden + head * head_dim;
                let dest = s * head_dim;
                q_head[dest..dest + head_dim].copy_from_slice(&q[source..source + head_dim]);
                k_head[dest..dest + head_dim].copy_from_slice(&k[source..source + head_dim]);
                v_head[dest..dest + head_dim].copy_from_slice(&v[source..source + head_dim]);
            }

            provider.matmul(
                seq,
                seq,
                head_dim,
                &q_head,
                &k_head,
                BLayout::RowMajorNkTransposed,
                &mut scores,
            )?;
            for query_pos in 0..seq {
                let row_start = query_pos * seq;
                let row = &mut scores[row_start..row_start + seq];
                for key_pos in 0..seq {
                    row[key_pos] *= scale;
                    if attention_mask[b * seq + key_pos] == 0 {
                        row[key_pos] = -10_000.0;
                    }
                }
                softmax(row);
            }

            provider.matmul(
                seq,
                head_dim,
                seq,
                &scores,
                &v_head,
                BLayout::RowMajorKn,
                &mut ctx_head,
            )?;
            for s in 0..seq {
                let source = s * head_dim;
                let dest = (b * seq + s) * hidden + head * head_dim;
                context[dest..dest + head_dim]
                    .copy_from_slice(&ctx_head[source..source + head_dim]);
            }
        }
    }
    Ok(context)
}

fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    let inv_sum = 1.0 / sum.max(1e-20);
    for value in row {
        *value *= inv_sum;
    }
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn mean_pool_l2(
    hidden: &Tensor,
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    hidden_size: usize,
) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut vector = vec![0.0f32; hidden_size];
        let mut count = 0.0f32;
        for s in 0..seq {
            if attention_mask[b * seq + s] == 1 {
                count += 1.0;
                let start = (b * seq + s) * hidden_size;
                for (value, hidden_value) in vector
                    .iter_mut()
                    .zip(&hidden.data[start..start + hidden_size])
                {
                    *value += *hidden_value;
                }
            }
        }
        let denom = count.max(1.0);
        for value in &mut vector {
            *value /= denom;
        }
        normalize_l2(&mut vector);
        vectors.push(vector);
    }
    vectors
}

fn normalize_l2(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    for value in vector {
        *value /= norm;
    }
}

fn load_safetensor_map(model_root: &Path, original_path: &Path) -> Result<HashMap<String, Tensor>> {
    if original_path.is_file()
        && original_path.extension().and_then(|value| value.to_str()) == Some("safetensors")
    {
        return load_single_safetensors_file(original_path);
    }

    let single_file = model_root.join("model.safetensors");
    if single_file.is_file() {
        return load_single_safetensors_file(&single_file);
    }

    let index_file = model_root.join("model.safetensors.index.json");
    if index_file.is_file() {
        #[derive(Deserialize)]
        struct SafetensorsIndex {
            weight_map: HashMap<String, String>,
        }
        let index: SafetensorsIndex = serde_json::from_str(
            &fs::read_to_string(&index_file)
                .with_context(|| format!("read safetensors index {}", index_file.display()))?,
        )
        .with_context(|| format!("parse safetensors index {}", index_file.display()))?;
        let mut merged = HashMap::new();
        let unique_files: HashSet<_> = index.weight_map.into_values().collect();
        for shard in unique_files {
            let shard_path = model_root.join(&shard);
            merged.extend(load_single_safetensors_file(&shard_path)?);
        }
        return Ok(merged);
    }

    bail!(
        "could not find model.safetensors or model.safetensors.index.json under {}",
        model_root.display()
    )
}

fn load_single_safetensors_file(path: &Path) -> Result<HashMap<String, Tensor>> {
    let bytes = fs::read(path).with_context(|| format!("read safetensors {}", path.display()))?;
    let safetensors = SafeTensors::deserialize(&bytes)
        .map_err(|error| anyhow::anyhow!("load safetensors {}: {error}", path.display()))?;
    let mut tensors = HashMap::new();
    for name in safetensors.names() {
        let view = safetensors
            .tensor(name)
            .map_err(|error| anyhow::anyhow!("read tensor {name}: {error}"))?;
        if matches!(
            view.dtype(),
            SafeDtype::F32 | SafeDtype::F16 | SafeDtype::BF16
        ) {
            tensors.insert(
                name.to_string(),
                tensor_from_view(view.dtype(), view.shape(), view.data())?,
            );
        }
    }
    Ok(tensors)
}

fn tensor_from_view(dtype: SafeDtype, shape: &[usize], bytes: &[u8]) -> Result<Tensor> {
    let values = match dtype {
        SafeDtype::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk_exact length")))
            .collect(),
        SafeDtype::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                half::f16::from_bits(u16::from_le_bytes(
                    chunk.try_into().expect("chunk_exact length"),
                ))
                .to_f32()
            })
            .collect(),
        SafeDtype::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                half::bf16::from_bits(u16::from_le_bytes(
                    chunk.try_into().expect("chunk_exact length"),
                ))
                .to_f32()
            })
            .collect(),
        other => bail!("unsupported safetensor dtype {other:?}; expected f32/f16/bf16"),
    };
    let mut tensor = Tensor::new(shape.to_vec(), values)?;
    if matches!(dtype, SafeDtype::F16) {
        tensor.metal_f16_bits = Some(
            bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("chunk_exact length")))
                .collect(),
        );
    }
    Ok(tensor)
}

fn resolve_model_root(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
        return path
            .parent()
            .map(Path::to_path_buf)
            .context("model file has no parent directory");
    }
    bail!(
        "model path {} is neither a directory nor a safetensors file",
        path.display()
    )
}

fn get_tensor(tensors: &HashMap<String, Tensor>, base_name: &str) -> Result<Tensor> {
    let candidates = [
        base_name.to_string(),
        format!("bert.{base_name}"),
        format!("model.{base_name}"),
        format!("model.bert.{base_name}"),
    ];
    for candidate in &candidates {
        if let Some(tensor) = tensors.get(candidate) {
            return Ok(tensor.clone());
        }
    }
    bail!("missing tensor; tried {}", candidates.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_execution_defaults_to_explicit_and_lazy_is_opt_in() {
        let base = [
            "spike-unified-rt",
            "--model",
            "model",
            "--tokenizer",
            "tokenizer.json",
            "--corpus",
            "corpus.jsonl",
            "--out",
            "result.json",
        ];
        let default = Args::try_parse_from(base).expect("parse default serving arguments");
        assert_eq!(default.execution, Execution::Explicit);
        let lazy = Args::try_parse_from(base.into_iter().chain(["--execution", "lazy"]))
            .expect("parse lazy execution override");
        assert_eq!(lazy.execution, Execution::Lazy);
    }

    #[test]
    fn package_cache_uses_one_path_per_shape() {
        let config = MetalExecutionConfig {
            execution: Execution::Explicit,
            package_root: Some(PathBuf::from("cache/model-graph-v2-f16-os")),
        };
        assert_eq!(
            config.package_path(8, 128),
            Some(PathBuf::from(
                "cache/model-graph-v2-f16-os/8x128.mpsgraphpackage"
            ))
        );
        assert_ne!(config.package_path(8, 128), config.package_path(4, 256));
    }

    #[test]
    fn qwen3_parity_gate_enforces_certification_thresholds() {
        enforce_parity_gates(0.9999, 0.995, 0.9999, 0.995, 400, 400)
            .expect("Qwen3 certification boundary must pass");
        assert!(enforce_parity_gates(0.99989, 1.0, 0.9999, 0.995, 400, 400).is_err());
        assert!(enforce_parity_gates(1.0, 0.9949, 0.9999, 0.995, 400, 400).is_err());
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_close_with_tolerance(actual, expected, 1e-3);
    }

    fn assert_close_with_tolerance(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff <= tolerance,
                "value {index} differs: actual={left}, expected={right}, diff={diff}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_provider_matches_cpu_for_row_major_rhs() {
        let mut metal = MetalProvider::new(Precision::F32).expect("create MPSGraph provider");
        let mut cpu = CpuProvider;
        let a = vec![1.0, 2.0, 3.0, 4.0, -2.0, 0.5];
        let b = vec![
            0.5, -1.0, 2.0, 1.5, 3.0, -0.5, -2.0, 0.25, 1.25, -1.5, 0.75, 2.5,
        ];
        let mut metal_out = vec![0.0; 8];
        let mut cpu_out = vec![0.0; 8];
        metal
            .matmul(2, 4, 3, &a, &b, BLayout::RowMajorKn, &mut metal_out)
            .expect("run MPSGraph matmul");
        cpu.matmul(2, 4, 3, &a, &b, BLayout::RowMajorKn, &mut cpu_out)
            .expect("run CPU matmul");
        assert_close(&metal_out, &cpu_out);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_provider_matches_cpu_for_transposed_rhs_storage() {
        let mut metal = MetalProvider::new(Precision::F32).expect("create MPSGraph provider");
        let mut cpu = CpuProvider;
        let a = vec![1.0, 2.0, 3.0, 4.0, -2.0, 0.5];
        let b = vec![
            0.5, 3.0, 1.25, -1.0, -0.5, -1.5, 2.0, -2.0, 0.75, 1.5, 0.25, 2.5,
        ];
        let mut metal_out = vec![0.0; 8];
        let mut cpu_out = vec![0.0; 8];
        metal
            .matmul(
                2,
                4,
                3,
                &a,
                &b,
                BLayout::RowMajorNkTransposed,
                &mut metal_out,
            )
            .expect("run MPSGraph matmul");
        cpu.matmul(2, 4, 3, &a, &b, BLayout::RowMajorNkTransposed, &mut cpu_out)
            .expect("run CPU matmul");
        assert_close(&metal_out, &cpu_out);
    }

    fn patterned_values(len: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((((index * 37) % 23) as f32) - 11.0) * scale + bias)
            .collect()
    }

    fn test_linear(output: usize, input: usize, scale: f32) -> Linear {
        Linear {
            weight: Tensor::new(
                vec![output, input],
                patterned_values(output * input, scale, 0.0),
            )
            .expect("linear test weight"),
            bias: ParamVector::new(patterned_values(output, scale * 0.25, 0.0)),
        }
    }

    fn test_layer(hidden: usize, intermediate: usize) -> EncoderLayer {
        EncoderLayer {
            query: test_linear(hidden, hidden, 0.011),
            key: test_linear(hidden, hidden, -0.009),
            value: test_linear(hidden, hidden, 0.007),
            attention_output: test_linear(hidden, hidden, 0.013),
            attention_ln_weight: ParamVector::new(patterned_values(hidden, 0.01, 1.0)),
            attention_ln_bias: ParamVector::new(patterned_values(hidden, 0.003, 0.0)),
            intermediate: test_linear(intermediate, hidden, 0.008),
            output: test_linear(hidden, intermediate, -0.006),
            output_ln_weight: ParamVector::new(patterned_values(hidden, 0.012, 1.0)),
            output_ln_bias: ParamVector::new(patterned_values(hidden, -0.002, 0.0)),
        }
    }

    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn run_minilm_block(
        provider: &mut MetalProvider,
        hidden_states: &mut [f32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
        hidden: usize,
        heads: usize,
        intermediate: usize,
        layers: &[EncoderLayer],
    ) -> Result<bool> {
        let mut run = |context: &mut dyn Any| {
            let context = context
                .downcast_mut::<MiniLmBlockContext>()
                .context("test provider returned the wrong MiniLM context")?;
            context.graph.encoder_forward(
                hidden_states,
                attention_mask,
                batch,
                seq,
                hidden,
                heads,
                intermediate,
                1e-12,
                layers,
                context.precision,
            )
        };
        provider.block_forward(BlockForwardRequest {
            family: "minilm",
            create_context: new_minilm_block_context,
            run: &mut run,
        })
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_provider_matches_cpu_for_tiny_encoder_block() {
        let batch = 2;
        let seq = 3;
        let hidden = 4;
        let heads = 2;
        let intermediate = 8;
        let attention_mask = vec![1, 1, 0, 1, 1, 1];
        let layers = vec![test_layer(hidden, intermediate)];
        let mut expected = patterned_values(batch * seq * hidden, 0.02, 0.01);
        let mut actual = expected.clone();

        let mut cpu = CpuProvider;
        encoder_layers_scalar_forward(
            &mut cpu,
            &mut expected,
            &attention_mask,
            batch,
            seq,
            hidden,
            heads,
            intermediate,
            1e-12,
            &layers,
        )
        .expect("run CPU encoder block");

        let mut metal = MetalProvider::new(Precision::F32).expect("create MPSGraph provider");
        assert!(run_minilm_block(
            &mut metal,
            &mut actual,
            &attention_mask,
            batch,
            seq,
            hidden,
            heads,
            intermediate,
            &layers,
        )
        .expect("run resident MPSGraph encoder block"));
        assert_close_with_tolerance(&actual, &expected, 5e-3);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_provider_f16_tracks_cpu_for_tiny_encoder_block() {
        let batch = 2;
        let seq = 3;
        let hidden = 4;
        let heads = 2;
        let intermediate = 8;
        let attention_mask = vec![1, 1, 0, 1, 1, 1];
        let mut layers = vec![test_layer(hidden, intermediate)];
        for layer in &mut layers {
            layer.prepare_metal_f16();
        }
        let mut expected = patterned_values(batch * seq * hidden, 0.02, 0.01);
        let mut actual = expected.clone();

        let mut cpu = CpuProvider;
        encoder_layers_scalar_forward(
            &mut cpu,
            &mut expected,
            &attention_mask,
            batch,
            seq,
            hidden,
            heads,
            intermediate,
            1e-12,
            &layers,
        )
        .expect("run CPU encoder block");

        let mut metal = MetalProvider::new(Precision::F16).expect("create MPSGraph provider");
        assert!(run_minilm_block(
            &mut metal,
            &mut actual,
            &attention_mask,
            batch,
            seq,
            hidden,
            heads,
            intermediate,
            &layers,
        )
        .expect("run resident MPSGraph encoder block"));
        assert_close_with_tolerance(&actual, &expected, 3e-2);
    }
}
