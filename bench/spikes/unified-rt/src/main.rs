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
    parity::{load_corpus, load_reference, mean_parity, Chunk},
    results::LaneResult,
};
use tokenizers::{Tokenizer, TruncationParams};

#[derive(Parser)]
#[command(name = "spike-unified-rt")]
struct Args {
    /// Path to a MiniLM safetensors file or snapshot directory.
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
    /// Kernel provider to use.
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,
    /// Tokenizer truncation max length.
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Greedy attention-unit budget per batch.
    #[arg(long, default_value_t = 4_000_000)]
    attention_units: usize,
    /// Model label for the result.
    #[arg(long, default_value = "all-MiniLM-L6-v2@owned-rt-fp32")]
    model_label: String,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum)]
enum DeviceArg {
    Cpu,
    Metal,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let started = Instant::now();

    let model = BertModel::load(&args.model)?;
    let mut tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: args.max_length,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!("truncation: {error}"))?;

    let mut provider = make_provider(args.device)?;
    let _ = model.embed_batch(provider.as_mut(), &tokenizer, &["warmup"])?;
    let cold_load_s = started.elapsed().as_secs_f64();

    let chunks: Vec<Chunk> = load_corpus(&args.corpus, args.limit)?;
    let all_texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    let encodings = tokenizer
        .encode_batch(all_texts, true)
        .map_err(|error| anyhow::anyhow!("encode_batch: {error}"))?;
    let lengths: Vec<usize> = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .collect();

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
            let vectors = model.embed_batch(provider.as_mut(), &tokenizer, &batch_texts)?;
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
            ensure!(
                mean >= args.min_parity,
                "mean parity {mean:.8} below minimum {:.8} over {matched} vectors",
                args.min_parity
            );
            Some(mean)
        }
        None => None,
    };

    let result = LaneResult {
        lane: format!("owned-rt-{}", provider.name()),
        workload: "embed-corpus-v1".into(),
        model: args.model_label,
        cold_load_s,
        infer_wall_s,
        input_tokens,
        tok_per_s: input_tokens as f64 / infer_wall_s,
        items,
        parity_mean_cosine,
        self_peak_rss_bytes: None,
        notes: format!(
            "direct BERT encoder, provider={}, length-sorted attention_units={}, max_len={}, mean_pool+l2 on CPU; providers may override the encoder block, and the Metal provider keeps encoder-layer hidden states resident inside one MPSGraph per batch shape",
            provider.name(),
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

fn make_provider(device: DeviceArg) -> Result<Box<dyn KernelProvider>> {
    match device {
        DeviceArg::Cpu => Ok(Box::new(CpuProvider)),
        DeviceArg::Metal => Ok(Box::new(MetalProvider::new()?)),
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

    #[allow(clippy::too_many_arguments)]
    fn encoder_forward(
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
    ) -> Result<bool> {
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
}

impl MetalProvider {
    fn new() -> Result<Self> {
        Ok(Self {
            context: metal_backend::MpsGraphContext::new()?,
        })
    }
}

impl KernelProvider for MetalProvider {
    fn name(&self) -> &'static str {
        "metal-mpsgraph-resident-encoder"
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
        self.context.matmul(m, n, k, a, b, b_layout, c, false)
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
        self.context.matmul(m, n, k, a, b, b_layout, c, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn encoder_forward(
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
    ) -> Result<bool> {
        ensure!(
            hidden_states.len() == batch * seq * hidden,
            "encoder hidden shape mismatch"
        );
        ensure!(
            attention_mask.len() == batch * seq,
            "encoder mask shape mismatch"
        );
        self.context.encoder_forward(
            hidden_states,
            attention_mask,
            batch,
            seq,
            hidden,
            heads,
            intermediate,
            layer_norm_eps,
            layers,
        )?;
        Ok(true)
    }
}

#[cfg(target_os = "macos")]
mod metal_backend {
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::NonNull;

    use anyhow::{bail, ensure, Result};

    use super::{BLayout, EncoderLayer};

    #[repr(C)]
    struct SynapseMpsEncoderLayerParams {
        query_weight: *const f32,
        query_bias: *const f32,
        key_weight: *const f32,
        key_bias: *const f32,
        value_weight: *const f32,
        value_bias: *const f32,
        attention_output_weight: *const f32,
        attention_output_bias: *const f32,
        attention_ln_weight: *const f32,
        attention_ln_bias: *const f32,
        intermediate_weight: *const f32,
        intermediate_bias: *const f32,
        output_weight: *const f32,
        output_bias: *const f32,
        output_ln_weight: *const f32,
        output_ln_bias: *const f32,
    }

    pub struct MpsGraphContext {
        raw: NonNull<c_void>,
    }

    impl MpsGraphContext {
        pub fn new() -> Result<Self> {
            let raw = unsafe { synapse_mps_context_new() };
            let raw = NonNull::new(raw).ok_or_else(last_error)?;
            Ok(Self { raw })
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
        ) -> Result<()> {
            let b_is_row_major_nk = match b_layout {
                BLayout::RowMajorKn => 0,
                BLayout::RowMajorNkTransposed => 1,
            };
            let status = unsafe {
                synapse_mps_matmul(
                    self.raw.as_ptr(),
                    m as u64,
                    n as u64,
                    k as u64,
                    a.as_ptr(),
                    b.as_ptr(),
                    b_is_row_major_nk,
                    c.as_mut_ptr(),
                    i32::from(cache_rhs),
                )
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
            let params: Vec<SynapseMpsEncoderLayerParams> = layers
                .iter()
                .map(|layer| SynapseMpsEncoderLayerParams {
                    query_weight: layer.query.weight.data.as_ptr(),
                    query_bias: layer.query.bias.as_ptr(),
                    key_weight: layer.key.weight.data.as_ptr(),
                    key_bias: layer.key.bias.as_ptr(),
                    value_weight: layer.value.weight.data.as_ptr(),
                    value_bias: layer.value.bias.as_ptr(),
                    attention_output_weight: layer.attention_output.weight.data.as_ptr(),
                    attention_output_bias: layer.attention_output.bias.as_ptr(),
                    attention_ln_weight: layer.attention_ln_weight.as_ptr(),
                    attention_ln_bias: layer.attention_ln_bias.as_ptr(),
                    intermediate_weight: layer.intermediate.weight.data.as_ptr(),
                    intermediate_bias: layer.intermediate.bias.as_ptr(),
                    output_weight: layer.output.weight.data.as_ptr(),
                    output_bias: layer.output.bias.as_ptr(),
                    output_ln_weight: layer.output_ln_weight.as_ptr(),
                    output_ln_bias: layer.output_ln_bias.as_ptr(),
                })
                .collect();
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
                    hidden_states.as_ptr(),
                    additive_mask.as_ptr(),
                    output.as_mut_ptr(),
                    params.as_ptr(),
                )
            };
            if status != 0 {
                bail!(
                    "MPSGraph encoder forward failed with status {status}: {}",
                    last_error()
                );
            }
            hidden_states.copy_from_slice(&output);
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
            a: *const f32,
            b: *const f32,
            b_is_row_major_nk: i32,
            c: *mut f32,
            cache_rhs: i32,
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
            input: *const f32,
            additive_mask: *const f32,
            output: *mut f32,
            layers: *const SynapseMpsEncoderLayerParams,
        ) -> i32;
        fn synapse_mps_last_error() -> *const c_char;
    }
}

#[cfg(not(target_os = "macos"))]
mod metal_backend {
    use anyhow::{bail, Result};

    use super::{BLayout, EncoderLayer};

    pub struct MpsGraphContext;

    impl MpsGraphContext {
        pub fn new() -> Result<Self> {
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
}

#[derive(Copy, Clone, Debug)]
enum TensorDType {
    F32,
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
        })
    }

    fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>();
        Self {
            dtype: TensorDType::F32,
            strides: row_major_strides(&shape),
            shape,
            data: vec![0.0; len],
        }
    }

    fn dim(&self, index: usize) -> usize {
        self.shape[index]
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
    layer_norm_weight: Vec<f32>,
    layer_norm_bias: Vec<f32>,
}

struct EncoderLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    attention_ln_weight: Vec<f32>,
    attention_ln_bias: Vec<f32>,
    intermediate: Linear,
    output: Linear,
    output_ln_weight: Vec<f32>,
    output_ln_bias: Vec<f32>,
}

struct Linear {
    weight: Tensor,
    bias: Vec<f32>,
}

impl BertModel {
    fn load(path: &Path) -> Result<Self> {
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
            layer_norm_weight: get_tensor(&tensors, "embeddings.LayerNorm.weight")?
                .as_vector()?
                .to_vec(),
            layer_norm_bias: get_tensor(&tensors, "embeddings.LayerNorm.bias")?
                .as_vector()?
                .to_vec(),
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
                attention_ln_weight: get_tensor(
                    &tensors,
                    &format!("{prefix}.attention.output.LayerNorm.weight"),
                )?
                .as_vector()?
                .to_vec(),
                attention_ln_bias: get_tensor(
                    &tensors,
                    &format!("{prefix}.attention.output.LayerNorm.bias"),
                )?
                .as_vector()?
                .to_vec(),
                intermediate: Linear::load(&tensors, &format!("{prefix}.intermediate.dense"))?,
                output: Linear::load(&tensors, &format!("{prefix}.output.dense"))?,
                output_ln_weight: get_tensor(
                    &tensors,
                    &format!("{prefix}.output.LayerNorm.weight"),
                )?
                .as_vector()?
                .to_vec(),
                output_ln_bias: get_tensor(&tensors, &format!("{prefix}.output.LayerNorm.bias"))?
                    .as_vector()?
                    .to_vec(),
            });
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
            &self.embeddings.layer_norm_weight,
            &self.embeddings.layer_norm_bias,
            self.config.layer_norm_eps,
        )?;

        if provider.encoder_forward(
            &mut x.data,
            attention_mask,
            batch,
            seq,
            hidden,
            heads,
            self.config.intermediate_size,
            self.config.layer_norm_eps,
            &self.layers,
        )? {
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
            &layer.attention_ln_weight,
            &layer.attention_ln_bias,
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
            &layer.output_ln_weight,
            &layer.output_ln_bias,
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
        let bias = get_tensor(tensors, &format!("{prefix}.bias"))?
            .as_vector()?
            .to_vec();
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
                out[start + col] += self.bias[col];
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
    Tensor::new(shape.to_vec(), values)
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
        let mut metal = MetalProvider::new().expect("create MPSGraph provider");
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
        let mut metal = MetalProvider::new().expect("create MPSGraph provider");
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
            bias: patterned_values(output, scale * 0.25, 0.0),
        }
    }

    fn test_layer(hidden: usize, intermediate: usize) -> EncoderLayer {
        EncoderLayer {
            query: test_linear(hidden, hidden, 0.011),
            key: test_linear(hidden, hidden, -0.009),
            value: test_linear(hidden, hidden, 0.007),
            attention_output: test_linear(hidden, hidden, 0.013),
            attention_ln_weight: patterned_values(hidden, 0.01, 1.0),
            attention_ln_bias: patterned_values(hidden, 0.003, 0.0),
            intermediate: test_linear(intermediate, hidden, 0.008),
            output: test_linear(hidden, intermediate, -0.006),
            output_ln_weight: patterned_values(hidden, 0.012, 1.0),
            output_ln_bias: patterned_values(hidden, -0.002, 0.0),
        }
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

        let mut metal = MetalProvider::new().expect("create MPSGraph provider");
        assert!(metal
            .encoder_forward(
                &mut actual,
                &attention_mask,
                batch,
                seq,
                hidden,
                heads,
                intermediate,
                1e-12,
                &layers,
            )
            .expect("run resident MPSGraph encoder block"));
        assert_close_with_tolerance(&actual, &expected, 5e-3);
    }
}
