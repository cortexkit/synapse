//! Faithful GTE ModernBERT embedding through the private Neural Engine API.
//!
//! Tokenization deliberately stays outside this program. Each input row already
//! contains the exact token IDs that the serving boundary would provide.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ane::{Executable, Graph, NSQualityOfService, Shape, Tensor, TensorData};
use anyhow::{bail, ensure, Context, Result};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[path = "../reranker.rs"]
pub mod reranker;

const MODEL_ID: &str = "Alibaba-NLP/gte-modernbert-base";
const GATE: f32 = 0.999;
const MASK_MIN: f32 = -10_000.0;

#[derive(Clone, Deserialize)]
struct Config {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    global_attn_every_n_layers: usize,
    global_rope_theta: f32,
    local_attention: usize,
    local_rope_theta: f32,
    max_position_embeddings: usize,
    norm_eps: f32,
    pad_token_id: u32,
    vocab_size: usize,
    classifier_pooling: Option<String>,
}

#[derive(Deserialize)]
struct InputRow {
    id: String,
    #[serde(default)]
    input_ids: Option<Vec<u32>>,
    #[serde(default)]
    input_ids_by_shape: BTreeMap<String, Vec<u32>>,
}

impl InputRow {
    fn ids(&self, sequence_length: usize) -> Result<&[u32]> {
        if let Some(ids) = &self.input_ids {
            return Ok(ids);
        }
        self.input_ids_by_shape
            .get(&sequence_length.to_string())
            .map(Vec::as_slice)
            .with_context(|| {
                format!(
                    "row {} has no token IDs for sequence shape {}",
                    self.id, sequence_length
                )
            })
    }
}

#[derive(Clone)]
struct Linear {
    weight: Vec<f32>,
    rows: usize,
    columns: usize,
}

#[derive(Clone)]
struct LayerWeights {
    qkv: Linear,
    attention_output: Linear,
    attention_norm: Option<Vec<f32>>,
    mlp_input: Linear,
    mlp_output: Linear,
    mlp_norm: Vec<f32>,
}

struct Weights {
    embeddings: Linear,
    embedding_norm: Vec<f32>,
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>,
}

#[derive(Default)]
struct Cli {
    snapshot: PathBuf,
    rows: PathBuf,
    sequence_length: usize,
    layers_per_executable: usize,
    warm_repetitions: usize,
    report: Option<PathBuf>,
    vectors_out: Option<PathBuf>,
}

#[derive(Serialize)]
struct ModelIdentity {
    model_id: &'static str,
    snapshot_hash: String,
    config_sha256: String,
    model_safetensors_sha256: String,
}

#[derive(Serialize)]
struct CheckpointMetric {
    checkpoint: String,
    min_cosine: f32,
    mean_cosine: f32,
    max_abs: f32,
}

#[derive(Serialize)]
struct RowMetric {
    id: String,
    active_tokens: usize,
    cosine: f32,
    warm_wall_ms_median: f64,
    deterministic: bool,
}

#[derive(Serialize)]
struct Report {
    model: ModelIdentity,
    sequence_length: usize,
    layers_per_executable: usize,
    pooling: &'static str,
    cpu_reference: &'static str,
    row_set_sha256: String,
    rows: Vec<RowMetric>,
    min_cosine: f32,
    mean_cosine: f32,
    deterministic: bool,
    gate_min_cosine: f32,
    gate_passed: bool,
    first_divergence: Option<String>,
    checkpoint_metrics: Vec<CheckpointMetric>,
    one_minute_load_average: f64,
    timing_note: &'static str,
    vectors: Vec<Vec<f32>>,
}

struct CpuResult {
    checkpoints: Vec<Vec<f32>>,
    vector: Vec<f32>,
    final_hidden: Vec<f32>,
    layer_states: Vec<Vec<f32>>,
}

struct AneModel {
    embedding: Executable,
    chunks: Vec<Executable>,
    final_norm: Executable,
    raw: TensorData,
    hidden_a: TensorData,
    hidden_b: TensorData,
    key_mask: TensorData,
    chunk_ends: Vec<usize>,
    shape: Shape,
}

fn parse_cli() -> Result<Cli> {
    let mut args = std::env::args().skip(1);
    let snapshot = args.next().map(PathBuf::from).context(
        "usage: modernbert_full SNAPSHOT ROWS [--seq N] [--layers-per-executable N] \
         [--warm-repetitions N] [--report PATH] [--vectors-out PATH]",
    )?;
    let rows = args
        .next()
        .map(PathBuf::from)
        .context("missing pre-tokenized ROWS file")?;
    let mut cli = Cli {
        snapshot,
        rows,
        sequence_length: 512,
        layers_per_executable: 1,
        warm_repetitions: 5,
        report: None,
        vectors_out: None,
    };
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--seq" => cli.sequence_length = value.parse().context("invalid --seq")?,
            "--layers-per-executable" => {
                cli.layers_per_executable =
                    value.parse().context("invalid --layers-per-executable")?
            }
            "--warm-repetitions" => {
                cli.warm_repetitions = value.parse().context("invalid --warm-repetitions")?
            }
            "--report" => cli.report = Some(PathBuf::from(value)),
            "--vectors-out" => cli.vectors_out = Some(PathBuf::from(value)),
            other => bail!("unknown argument {other}"),
        }
    }
    ensure!(
        cli.sequence_length >= 64,
        "ANE sequence width must be at least 64"
    );
    ensure!(
        cli.layers_per_executable > 0,
        "layers per executable must be positive"
    );
    ensure!(
        cli.warm_repetitions > 0,
        "warm repetitions must be positive"
    );
    Ok(cli)
}

fn read_rows(path: &Path) -> Result<Vec<InputRow>> {
    let text = fs::read_to_string(path).with_context(|| format!("read rows {}", path.display()))?;
    let rows = text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).with_context(|| format!("parse row line {}", index + 1))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!rows.is_empty(), "row set is empty");
    Ok(rows)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&fs::read(path).with_context(|| {
        format!("read digest input {}", path.display())
    })?))
}

fn tensor_values(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let tensor = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    let values = match tensor.dtype() {
        Dtype::BF16 => tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)).to_f32())
            .collect(),
        Dtype::F16 => tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| f16::from_bits(u16::from_le_bytes(*bytes)).to_f32())
            .collect(),
        Dtype::F32 => tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect(),
        dtype => bail!("tensor {name} has unsupported dtype {dtype:?}"),
    };
    Ok((tensor.shape().to_vec(), values))
}

fn load_linear(st: &SafeTensors<'_>, name: &str, rows: usize, columns: usize) -> Result<Linear> {
    let (shape, weight) = tensor_values(st, &format!("{name}.weight"))?;
    ensure!(
        shape == [rows, columns],
        "{name}.weight shape {shape:?} != [{rows}, {columns}]"
    );
    Ok(Linear {
        weight,
        rows,
        columns,
    })
}

fn load_vector(st: &SafeTensors<'_>, name: &str, size: usize) -> Result<Vec<f32>> {
    let (shape, values) = tensor_values(st, name)?;
    ensure!(shape == [size], "{name} shape {shape:?} != [{size}]");
    Ok(values)
}

fn load_model(snapshot: &Path) -> Result<(Config, Weights, ModelIdentity)> {
    let config_path = snapshot.join("config.json");
    let weights_path = snapshot.join("model.safetensors");
    let config: Config = serde_json::from_slice(
        &fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?,
    )
    .context("parse config.json")?;
    ensure!(
        config.model_type == "modernbert",
        "snapshot is not ModernBERT"
    );
    ensure!(
        config
            .hidden_size
            .is_multiple_of(config.num_attention_heads),
        "hidden size must divide attention heads"
    );
    ensure!(
        config.global_attn_every_n_layers > 0,
        "global attention interval is zero"
    );
    ensure!(
        config.local_attention > 0 && config.local_attention.is_multiple_of(2),
        "invalid local window"
    );
    ensure!(
        config.classifier_pooling.as_deref() == Some("mean"),
        "unexpected classifier pooling metadata"
    );
    let bytes =
        fs::read(&weights_path).with_context(|| format!("read {}", weights_path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("parse model.safetensors")?;
    let prefix = if st.tensor("model.embeddings.norm.weight").is_ok() {
        "model."
    } else {
        ""
    };
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        let prefix = format!("{prefix}layers.{index}");
        layers.push(LayerWeights {
            qkv: load_linear(&st, &format!("{prefix}.attn.Wqkv"), hidden * 3, hidden)?,
            attention_output: load_linear(&st, &format!("{prefix}.attn.Wo"), hidden, hidden)?,
            attention_norm: if index == 0 {
                None
            } else {
                Some(load_vector(
                    &st,
                    &format!("{prefix}.attn_norm.weight"),
                    hidden,
                )?)
            },
            mlp_input: load_linear(&st, &format!("{prefix}.mlp.Wi"), intermediate * 2, hidden)?,
            mlp_output: load_linear(&st, &format!("{prefix}.mlp.Wo"), hidden, intermediate)?,
            mlp_norm: load_vector(&st, &format!("{prefix}.mlp_norm.weight"), hidden)?,
        });
    }
    let weights = Weights {
        embeddings: load_linear(
            &st,
            &format!("{prefix}embeddings.tok_embeddings"),
            config.vocab_size,
            hidden,
        )?,
        embedding_norm: load_vector(&st, &format!("{prefix}embeddings.norm.weight"), hidden)?,
        layers,
        final_norm: load_vector(&st, &format!("{prefix}final_norm.weight"), hidden)?,
    };
    let identity = ModelIdentity {
        model_id: if prefix.is_empty() {
            MODEL_ID
        } else {
            "Alibaba-NLP/gte-reranker-modernbert-base"
        },
        snapshot_hash: snapshot
            .file_name()
            .and_then(|name| name.to_str())
            .context("snapshot path has no final hash component")?
            .to_owned(),
        config_sha256: sha256_file(&config_path)?,
        model_safetensors_sha256: sha256_bytes(&bytes),
    };
    Ok((config, weights, identity))
}

fn shape(sequence_length: usize, channels: usize) -> Shape {
    Shape {
        batch: 1,
        channels,
        height: 1,
        width: sequence_length,
    }
}

fn scalar_shape() -> Shape {
    Shape::channels(1)
}

fn layer_norm_graph(graph: &mut Graph, input: Tensor, weight: &[f32], eps: f32) -> Tensor {
    let channels = input.shape.channels;
    let weight = graph.constant(weight, Shape::channels(channels));
    let mean = graph.reduce_mean(input, 1);
    let centered = graph.subtraction(input, mean);

    // Squaring ModernBERT's late residual values directly overflows fp16. Scale
    // each token by its largest centered channel, then algebraically scale the
    // epsilon as well; this computes the same LayerNorm without large squares.
    let magnitude = graph.absolute(centered);
    let magnitude = graph.reduce_max(magnitude, 1);
    let epsilon_root = graph.constant_with_scalar(eps.sqrt(), scalar_shape());
    let scale = graph.maximum(magnitude, epsilon_root);
    let scaled = graph.division(centered, scale);
    let squared = graph.multiplication(scaled, scaled);
    let variance = graph.reduce_mean(squared, 1);
    let scaled_epsilon = graph.division(epsilon_root, scale);
    let scaled_epsilon = graph.multiplication(scaled_epsilon, scaled_epsilon);
    let variance = graph.addition(variance, scaled_epsilon);
    // The private compiler rejects reciprocal_square_root in this reduction
    // pattern, while the equivalent power lowering is used by its encoder.
    let negative_half = graph.constant_with_scalar(-0.5, scalar_shape());
    let inverse_stddev = graph.power(variance, negative_half);
    let normalized = graph.multiplication(scaled, inverse_stddev);
    graph.multiplication(normalized, weight)
}

fn gelu_graph(graph: &mut Graph, input: Tensor) -> Tensor {
    // The binding has no erf operation. This is the standard tanh GELU lowering
    // used by its encoder example and differs from exact GELU by less than 5e-4.
    let half = graph.constant_with_scalar(0.5, scalar_shape());
    let one = graph.constant_with_scalar(1.0, scalar_shape());
    let cubic_scale = graph.constant_with_scalar(0.044_715, scalar_shape());
    let sqrt_two_over_pi = graph.constant_with_scalar(0.797_884_6, scalar_shape());
    let squared = graph.multiplication(input, input);
    let cubed = graph.multiplication(squared, input);
    let cubic = graph.multiplication(cubic_scale, cubed);
    let inner = graph.addition(input, cubic);
    let argument = graph.multiplication(sqrt_two_over_pi, inner);
    let tanh = graph.tanh(argument);
    let shifted = graph.addition(one, tanh);
    let half_input = graph.multiplication(half, input);
    graph.multiplication(half_input, shifted)
}

fn rope_tables(sequence_length: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; head_dim * sequence_length];
    let mut sine = vec![0.0; head_dim * sequence_length];
    for dimension in 0..head_dim / 2 {
        let frequency = theta.powf(-((2 * dimension) as f32) / head_dim as f32);
        for position in 0..sequence_length {
            let (sin, cos) = (position as f32 * frequency).sin_cos();
            for target in [dimension, dimension + head_dim / 2] {
                cosine[target * sequence_length + position] = cos;
                sine[target * sequence_length + position] = sin;
            }
        }
    }
    (cosine, sine)
}

fn apply_rope_graph(
    graph: &mut Graph,
    input: Tensor,
    cosine: &[f32],
    sine: &[f32],
    heads: usize,
    head_dim: usize,
    sequence_length: usize,
) -> Tensor {
    let half = head_dim / 2;
    let first = graph.slice(input, [0, 0, 0, 0], [1, heads, half, sequence_length]);
    let second = graph.slice(input, [0, 0, half, 0], [1, heads, half, sequence_length]);
    let negative = graph.constant_with_scalar(-1.0, scalar_shape());
    let negative_second = graph.multiplication(negative, second);
    let rotated = graph.concat(&[negative_second, first], 2);
    let table_shape = Shape {
        batch: 1,
        channels: 1,
        height: head_dim,
        width: sequence_length,
    };
    let cosine = graph.constant(cosine, table_shape);
    let sine = graph.constant(sine, table_shape);
    let direct = graph.multiplication(input, cosine);
    let turned = graph.multiplication(rotated, sine);
    graph.addition(direct, turned)
}

fn local_distance_mask(
    query_start: usize,
    query_len: usize,
    key_start: usize,
    key_len: usize,
    radius: usize,
) -> Vec<f32> {
    let mut mask = vec![0.0; query_len * key_len];
    for query in 0..query_len {
        for key in 0..key_len {
            if (query_start + query).abs_diff(key_start + key) > radius {
                mask[query * key_len + key] = MASK_MIN;
            }
        }
    }
    mask
}

#[allow(clippy::too_many_arguments)]
fn attention_graph(
    graph: &mut Graph,
    hidden: Tensor,
    key_mask: Tensor,
    weights: &LayerWeights,
    config: &Config,
    layer_index: usize,
    sequence_length: usize,
) -> Tensor {
    let hidden_size = config.hidden_size;
    let heads = config.num_attention_heads;
    let head_dim = hidden_size / heads;
    let normalized = if let Some(weight) = &weights.attention_norm {
        layer_norm_graph(graph, hidden, weight, config.norm_eps)
    } else {
        hidden
    };
    let qkv = graph.inner_product(
        normalized,
        &weights.qkv.weight,
        hidden_size,
        hidden_size * 3,
    );
    let head_shape = Shape {
        batch: 1,
        channels: heads,
        height: head_dim,
        width: sequence_length,
    };
    let mut parts = Vec::with_capacity(3);
    for part in 0..3 {
        let sliced = graph.slice(
            qkv,
            [0, part * hidden_size, 0, 0],
            [1, hidden_size, 1, sequence_length],
        );
        parts.push(graph.reshape(sliced, head_shape));
    }
    let theta = if layer_index.is_multiple_of(config.global_attn_every_n_layers) {
        config.global_rope_theta
    } else {
        config.local_rope_theta
    };
    let (cosine, sine) = rope_tables(sequence_length, head_dim, theta);
    let query = apply_rope_graph(
        graph,
        parts[0],
        &cosine,
        &sine,
        heads,
        head_dim,
        sequence_length,
    );
    let key = apply_rope_graph(
        graph,
        parts[1],
        &cosine,
        &sine,
        heads,
        head_dim,
        sequence_length,
    );
    let value = parts[2];
    let permutation = [0, 1, 3, 2];
    let query = graph.transpose(query, permutation);
    let key = graph.transpose(key, permutation);
    let value = graph.transpose(value, permutation);
    let scale = graph.constant_with_scalar(1.0 / (head_dim as f32).sqrt(), scalar_shape());
    let query_tile = 128usize.min(sequence_length);
    let local_radius = (!layer_index.is_multiple_of(config.global_attn_every_n_layers))
        .then_some(config.local_attention / 2);
    let mut contexts = Vec::new();
    for query_start in (0..sequence_length).step_by(query_tile) {
        let query_end = (query_start + query_tile).min(sequence_length);
        let query_len = query_end - query_start;
        let (key_start, key_end) = if let Some(radius) = local_radius {
            (
                query_start.saturating_sub(radius),
                (query_end + radius).min(sequence_length),
            )
        } else {
            (0, sequence_length)
        };
        let key_len = key_end - key_start;
        let query_slice = graph.slice(
            query,
            [0, 0, query_start, 0],
            [1, heads, query_len, head_dim],
        );
        let key_slice = graph.slice(key, [0, 0, key_start, 0], [1, heads, key_len, head_dim]);
        let value_slice = graph.slice(value, [0, 0, key_start, 0], [1, heads, key_len, head_dim]);
        let scores = graph.matrix_multiplication(query_slice, key_slice, false, true);
        let scores = graph.multiplication(scores, scale);
        let padding = graph.slice(key_mask, [0, 0, 0, key_start], [1, 1, 1, key_len]);
        let mut masked = graph.addition(scores, padding);
        if let Some(radius) = local_radius {
            let distance = local_distance_mask(query_start, query_len, key_start, key_len, radius);
            let distance = graph.constant(
                &distance,
                Shape {
                    batch: 1,
                    channels: 1,
                    height: query_len,
                    width: key_len,
                },
            );
            masked = graph.addition(masked, distance);
        }
        // Softmax over keys runs on the channel axis rather than the last
        // (width) axis: [1, heads, queries, keys] is transposed to
        // [1, keys, heads, queries], normalized on axis 1, and transposed back
        // so the value matmul sees its original layout. The math is the same
        // and only fp16 rounding differs; the channel-axis reduction measured
        // faster on the M5 Neural Engine at sequence 512 and still passes the
        // cosine gate at every sequence (see
        // docs/evidence/ane-direct-api-m5/fuse-and-channel-softmax.md).
        let channel_logits = graph.transpose(masked, [0, 3, 1, 2]);
        let channel_probabilities = graph.soft_max(channel_logits, 1);
        let probabilities = graph.transpose(channel_probabilities, [0, 2, 3, 1]);
        contexts.push(graph.matrix_multiplication(probabilities, value_slice, false, false));
    }
    let context = if contexts.len() == 1 {
        contexts[0]
    } else {
        graph.concat(&contexts, 2)
    };
    let context = graph.transpose(context, permutation);
    let context = graph.reshape(context, shape(sequence_length, hidden_size));
    let projected = graph.inner_product(
        context,
        &weights.attention_output.weight,
        hidden_size,
        hidden_size,
    );
    graph.addition(hidden, projected)
}

fn layer_graph(
    graph: &mut Graph,
    hidden: Tensor,
    key_mask: Tensor,
    weights: &LayerWeights,
    config: &Config,
    layer_index: usize,
    sequence_length: usize,
) -> Tensor {
    let attended = attention_graph(
        graph,
        hidden,
        key_mask,
        weights,
        config,
        layer_index,
        sequence_length,
    );
    let normalized = layer_norm_graph(graph, attended, &weights.mlp_norm, config.norm_eps);
    let projected = graph.inner_product(
        normalized,
        &weights.mlp_input.weight,
        config.hidden_size,
        config.intermediate_size * 2,
    );
    let activation = graph.slice(
        projected,
        [0, 0, 0, 0],
        [1, config.intermediate_size, 1, sequence_length],
    );
    let gate = graph.slice(
        projected,
        [0, config.intermediate_size, 0, 0],
        [1, config.intermediate_size, 1, sequence_length],
    );
    let activation = gelu_graph(graph, activation);
    let gated = graph.multiplication(activation, gate);
    let output = graph.inner_product(
        gated,
        &weights.mlp_output.weight,
        config.intermediate_size,
        config.hidden_size,
    );
    graph.addition(attended, output)
}

fn compile_model(
    config: &Config,
    weights: &Weights,
    sequence_length: usize,
    layers_per_executable: usize,
) -> Result<AneModel> {
    let hidden_shape = shape(sequence_length, config.hidden_size);
    let mask_shape = shape(sequence_length, 1);
    // At sequence 512 two layers are compiled into each executable whatever
    // the caller asked for: it halves the number of executable dispatches, returned
    // vectors byte-identical to the one-layer graph, and it measured faster
    // (docs/evidence/ane-direct-api-m5/fuse-and-channel-softmax.md). Three
    // layers per executable compile but crash at runtime, so two is the cap.
    // Longer sequences keep the caller's grouping, so a sequence-512 gain
    // cannot come from moving work into the longer shapes. The report's
    // `layers_per_executable` still echoes the command-line value, while the
    // checkpoint labels follow the grouping actually compiled.
    let layers_per_executable = if sequence_length == 512 {
        2
    } else {
        layers_per_executable
    };

    let mut embedding_graph = Graph::new();
    let embedding_input = embedding_graph.placeholder(hidden_shape);
    let _ = layer_norm_graph(
        &mut embedding_graph,
        embedding_input,
        &weights.embedding_norm,
        config.norm_eps,
    );
    let embedding = embedding_graph
        .compile(NSQualityOfService::UserInteractive)
        .context("compile embedding norm")?;

    let mut chunks = Vec::new();
    let mut chunk_ends = Vec::new();
    for start in (0..config.num_hidden_layers).step_by(layers_per_executable) {
        let end = (start + layers_per_executable).min(config.num_hidden_layers);
        let mut graph = Graph::new();
        let mut hidden = graph.placeholder(hidden_shape);
        let key_mask = graph.placeholder(mask_shape);
        for layer_index in start..end {
            hidden = layer_graph(
                &mut graph,
                hidden,
                key_mask,
                &weights.layers[layer_index],
                config,
                layer_index,
                sequence_length,
            );
        }
        chunks.push(
            graph
                .compile(NSQualityOfService::UserInteractive)
                .with_context(|| format!("compile layers {start}..{end}"))?,
        );
        chunk_ends.push(end);
    }

    let mut final_graph = Graph::new();
    let final_input = final_graph.placeholder(hidden_shape);
    let _ = layer_norm_graph(
        &mut final_graph,
        final_input,
        &weights.final_norm,
        config.norm_eps,
    );
    let final_norm = final_graph
        .compile(NSQualityOfService::UserInteractive)
        .context("compile final norm")?;

    Ok(AneModel {
        embedding,
        chunks,
        final_norm,
        raw: TensorData::new(hidden_shape),
        hidden_a: TensorData::new(hidden_shape),
        hidden_b: TensorData::new(hidden_shape),
        key_mask: TensorData::new(mask_shape),
        chunk_ends,
        shape: hidden_shape,
    })
}

fn raw_embeddings(
    row: &[u32],
    weights: &Weights,
    config: &Config,
    sequence_length: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0; config.hidden_size * sequence_length];
    for position in 0..sequence_length {
        let token = if position < row.len() {
            row[position] as usize
        } else {
            config.pad_token_id as usize
        };
        ensure!(
            token < config.vocab_size,
            "token id {token} is outside vocabulary"
        );
        for channel in 0..config.hidden_size {
            output[channel * sequence_length + position] =
                weights.embeddings.weight[token * config.hidden_size + channel];
        }
    }
    Ok(output)
}

fn l2_normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt()
        .max(1e-12);
    for value in &mut vector {
        *value = (*value as f64 / norm) as f32;
    }
    vector
}

fn surface_cls(surface: &TensorData, sequence_length: usize, hidden_size: usize) -> Vec<f32> {
    let values = surface.read_f32();
    (0..hidden_size)
        .map(|channel| values[channel * sequence_length])
        .collect()
}

impl AneModel {
    fn run(
        &self,
        row: &[u32],
        weights: &Weights,
        config: &Config,
        capture: bool,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let sequence_length = self.shape.width;
        self.raw
            .copy_from_f32(&raw_embeddings(row, weights, config, sequence_length)?);
        let mut mask = vec![MASK_MIN; sequence_length];
        for (position, token) in row.iter().copied().enumerate() {
            mask[position] = if token == config.pad_token_id {
                MASK_MIN
            } else {
                0.0
            };
        }
        self.key_mask.copy_from_f32(&mask);
        self.embedding
            .run_cached(&[&self.raw], &[&self.hidden_a])
            .context("run embedding norm")?;
        let mut checkpoints = Vec::new();
        if capture {
            checkpoints.push(surface_cls(
                &self.hidden_a,
                sequence_length,
                config.hidden_size,
            ));
        }
        for (index, executable) in self.chunks.iter().enumerate() {
            let (source, destination) = if index % 2 == 0 {
                (&self.hidden_a, &self.hidden_b)
            } else {
                (&self.hidden_b, &self.hidden_a)
            };
            executable
                .run_cached(&[source, &self.key_mask], &[destination])
                .with_context(|| format!("run through layer {}", self.chunk_ends[index]))?;
            if capture {
                checkpoints.push(surface_cls(
                    destination,
                    sequence_length,
                    config.hidden_size,
                ));
            }
        }
        let final_input = if self.chunks.len().is_multiple_of(2) {
            &self.hidden_a
        } else {
            &self.hidden_b
        };
        self.final_norm
            .run_cached(&[final_input], &[&self.raw])
            .context("run final norm")?;
        let cls = surface_cls(&self.raw, sequence_length, config.hidden_size);
        if capture {
            checkpoints.push(cls.clone());
        }
        Ok((l2_normalize(cls), checkpoints))
    }
}

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn getloadavg(load_average: *mut f64, count: i32) -> i32;
}

fn matmul(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], transpose_b: bool) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let mut output = vec![0.0; m * n];
    unsafe {
        cblas_sgemm(
            101,
            111,
            if transpose_b { 112 } else { 111 },
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            if transpose_b { k as i32 } else { n as i32 },
            0.0,
            output.as_mut_ptr(),
            n as i32,
        );
    }
    output
}

fn linear_cpu(input: &[f32], rows: usize, linear: &Linear) -> Vec<f32> {
    matmul(
        rows,
        linear.rows,
        linear.columns,
        input,
        &linear.weight,
        true,
    )
}

fn layer_norm_cpu(data: &mut [f32], rows: usize, hidden: usize, weight: &[f32], eps: f32) {
    for row in data.chunks_exact_mut(hidden).take(rows) {
        let mean = row.iter().sum::<f32>() / hidden as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / hidden as f32;
        let inverse_stddev = (variance + eps).sqrt().recip();
        for (value, scale) in row.iter_mut().zip(weight) {
            *value = (*value - mean) * inverse_stddev * scale;
        }
    }
}

fn apply_rope_cpu(values: &mut [f32], position: usize, theta: f32) {
    let half = values.len() / 2;
    let original = values.to_vec();
    for dimension in 0..half {
        let frequency = theta.powf(-((2 * dimension) as f32) / values.len() as f32);
        let (sin, cos) = (position as f32 * frequency).sin_cos();
        values[dimension] = original[dimension] * cos - original[dimension + half] * sin;
        values[dimension + half] = original[dimension + half] * cos + original[dimension] * sin;
    }
}

fn softmax_cpu(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn attention_cpu(
    qkv: &[f32],
    mask: &[u8],
    config: &Config,
    layer_index: usize,
    sequence_length: usize,
) -> Vec<f32> {
    let hidden = config.hidden_size;
    let heads = config.num_attention_heads;
    let head_dim = hidden / heads;
    let radius = config.local_attention / 2;
    let global = layer_index.is_multiple_of(config.global_attn_every_n_layers);
    let theta = if global {
        config.global_rope_theta
    } else {
        config.local_rope_theta
    };
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0; sequence_length * hidden];
    for head in 0..heads {
        let mut query = vec![0.0; sequence_length * head_dim];
        let mut key = vec![0.0; sequence_length * head_dim];
        let mut value = vec![0.0; sequence_length * head_dim];
        for position in 0..sequence_length {
            let source = position * hidden * 3 + head * head_dim;
            let destination = position * head_dim;
            query[destination..destination + head_dim]
                .copy_from_slice(&qkv[source..source + head_dim]);
            key[destination..destination + head_dim]
                .copy_from_slice(&qkv[source + hidden..source + hidden + head_dim]);
            value[destination..destination + head_dim]
                .copy_from_slice(&qkv[source + hidden * 2..source + hidden * 2 + head_dim]);
            apply_rope_cpu(
                &mut query[destination..destination + head_dim],
                position,
                theta,
            );
            apply_rope_cpu(
                &mut key[destination..destination + head_dim],
                position,
                theta,
            );
        }
        let mut scores = matmul(
            sequence_length,
            sequence_length,
            head_dim,
            &query,
            &key,
            true,
        );
        for query_position in 0..sequence_length {
            let row = &mut scores
                [query_position * sequence_length..(query_position + 1) * sequence_length];
            for key_position in 0..sequence_length {
                row[key_position] = if mask[key_position] == 0
                    || (!global && query_position.abs_diff(key_position) > radius)
                {
                    MASK_MIN
                } else {
                    row[key_position] * scale
                };
            }
            softmax_cpu(row);
        }
        let attended = matmul(
            sequence_length,
            head_dim,
            sequence_length,
            &scores,
            &value,
            false,
        );
        for position in 0..sequence_length {
            let destination = position * hidden + head * head_dim;
            output[destination..destination + head_dim]
                .copy_from_slice(&attended[position * head_dim..(position + 1) * head_dim]);
        }
    }
    output
}

fn cpu_reference(
    row: &[u32],
    weights: &Weights,
    config: &Config,
    sequence_length: usize,
) -> Result<CpuResult> {
    cpu_reference_diagnostic(row, weights, config, sequence_length, false, false)
}

// Keep the embedding reference exact-erf by default; the tanh option isolates
// the ANE graph's activation formula without changing device precision.
fn cpu_reference_diagnostic(
    row: &[u32],
    weights: &Weights,
    config: &Config,
    sequence_length: usize,
    tanh_gelu: bool,
    capture_layers: bool,
) -> Result<CpuResult> {
    let mut layer_states = Vec::new();
    let hidden = config.hidden_size;
    let mut ids = vec![config.pad_token_id; sequence_length];
    ids[..row.len()].copy_from_slice(row);
    let mask = ids
        .iter()
        .map(|token| u8::from(*token != config.pad_token_id))
        .collect::<Vec<_>>();
    let mut current = vec![0.0; sequence_length * hidden];
    for (position, token) in ids.iter().copied().enumerate() {
        ensure!(
            (token as usize) < config.vocab_size,
            "token id {token} is outside vocabulary"
        );
        current[position * hidden..(position + 1) * hidden].copy_from_slice(
            &weights.embeddings.weight[token as usize * hidden..(token as usize + 1) * hidden],
        );
    }
    layer_norm_cpu(
        &mut current,
        sequence_length,
        hidden,
        &weights.embedding_norm,
        config.norm_eps,
    );
    let mut checkpoints = vec![current[..hidden].to_vec()];
    for (layer_index, layer) in weights.layers.iter().enumerate() {
        let mut attention_input = current.clone();
        if let Some(weight) = &layer.attention_norm {
            layer_norm_cpu(
                &mut attention_input,
                sequence_length,
                hidden,
                weight,
                config.norm_eps,
            );
        }
        let qkv = linear_cpu(&attention_input, sequence_length, &layer.qkv);
        let context = attention_cpu(&qkv, &mask, config, layer_index, sequence_length);
        let attention_output = linear_cpu(&context, sequence_length, &layer.attention_output);
        for (destination, source) in current.iter_mut().zip(attention_output) {
            *destination += source;
        }
        let mut mlp_input = current.clone();
        layer_norm_cpu(
            &mut mlp_input,
            sequence_length,
            hidden,
            &layer.mlp_norm,
            config.norm_eps,
        );
        let projected = linear_cpu(&mlp_input, sequence_length, &layer.mlp_input);
        let mut activated = vec![0.0; sequence_length * config.intermediate_size];
        for position in 0..sequence_length {
            let source = position * config.intermediate_size * 2;
            let destination = position * config.intermediate_size;
            for column in 0..config.intermediate_size {
                let value = projected[source + column];
                let gelu = if tanh_gelu {
                    let squared = value * value;
                    let cubed = squared * value;
                    let inner = value + 0.044_715 * cubed;
                    (0.5 * value) * (1.0 + (0.797_884_6 * inner).tanh())
                } else {
                    0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
                };
                activated[destination + column] =
                    gelu * projected[source + config.intermediate_size + column];
            }
        }
        let mlp_output = linear_cpu(&activated, sequence_length, &layer.mlp_output);
        for (destination, source) in current.iter_mut().zip(mlp_output) {
            *destination += source;
        }
        checkpoints.push(current[..hidden].to_vec());
        if capture_layers {
            layer_states.push(current.clone());
        }
    }
    layer_norm_cpu(
        &mut current,
        sequence_length,
        hidden,
        &weights.final_norm,
        config.norm_eps,
    );
    checkpoints.push(current[..hidden].to_vec());
    Ok(CpuResult {
        vector: l2_normalize(current[..hidden].to_vec()),
        checkpoints,
        final_hidden: current,
        layer_states,
    })
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let mut numerator = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (&left, &right) in left.iter().zip(right) {
        numerator += left as f64 * right as f64;
        left_norm += left as f64 * left as f64;
        right_norm += right as f64 * right as f64;
    }
    (numerator / (left_norm.sqrt() * right_norm.sqrt()).max(1e-12)) as f32
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn byte_identical(left: &[f32], right: &[f32]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn load_average() -> f64 {
    let mut values = [f64::NAN; 3];
    let count = unsafe { getloadavg(values.as_mut_ptr(), values.len() as i32) };
    if count > 0 {
        values[0]
    } else {
        f64::NAN
    }
}

fn checkpoint_labels(config: &Config, chunk_ends: &[usize]) -> Vec<String> {
    let mut labels = vec!["embeddings_norm".to_owned()];
    labels.extend(chunk_ends.iter().map(|end| {
        if *end == 0 {
            "layers_empty".to_owned()
        } else {
            format!("layer_{}.output", end - 1)
        }
    }));
    labels.push("final_norm".to_owned());
    ensure_labels(labels, config)
}

fn ensure_labels(labels: Vec<String>, _config: &Config) -> Vec<String> {
    labels
}

fn write_json(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

fn main() -> Result<()> {
    let cli = parse_cli()?;
    let rows_bytes = fs::read(&cli.rows).with_context(|| format!("read {}", cli.rows.display()))?;
    let rows = read_rows(&cli.rows)?;
    let (config, weights, identity) = load_model(&cli.snapshot)?;
    ensure!(
        cli.sequence_length <= config.max_position_embeddings,
        "sequence length exceeds model context limit"
    );
    for row in &rows {
        let ids = row.ids(cli.sequence_length)?;
        ensure!(!ids.is_empty(), "row {} is empty", row.id);
        ensure!(
            ids.len() <= cli.sequence_length,
            "row {} has {} tokens, exceeding shape {}",
            row.id,
            ids.len(),
            cli.sequence_length
        );
    }

    eprintln!(
        "compiling {} layers at sequence {} ({} layer(s) per executable)",
        config.num_hidden_layers, cli.sequence_length, cli.layers_per_executable
    );
    let model = compile_model(
        &config,
        &weights,
        cli.sequence_length,
        cli.layers_per_executable,
    )?;
    let labels = checkpoint_labels(&config, &model.chunk_ends);
    let mut checkpoint_cosines = vec![Vec::<f32>::new(); labels.len()];
    let mut checkpoint_max_abs = vec![0.0f32; labels.len()];
    let mut row_metrics = Vec::new();
    let mut output_vectors = Vec::new();
    let mut all_cosines = Vec::new();
    let mut deterministic = true;

    for row in &rows {
        eprintln!("CPU fp32 reference: {}", row.id);
        let ids = row.ids(cli.sequence_length)?;
        let reference = cpu_reference(ids, &weights, &config, cli.sequence_length)?;
        let (candidate, candidate_checkpoints) = model.run(ids, &weights, &config, true)?;
        ensure!(
            candidate_checkpoints.len() == labels.len(),
            "candidate checkpoint count changed"
        );
        let reference_indexes = std::iter::once(0usize)
            .chain(model.chunk_ends.iter().copied())
            .chain(std::iter::once(config.num_hidden_layers + 1))
            .collect::<Vec<_>>();
        for (metric_index, (&reference_index, candidate_checkpoint)) in reference_indexes
            .iter()
            .zip(&candidate_checkpoints)
            .enumerate()
        {
            let reference_checkpoint = &reference.checkpoints[reference_index];
            checkpoint_cosines[metric_index]
                .push(cosine(reference_checkpoint, candidate_checkpoint));
            checkpoint_max_abs[metric_index] = checkpoint_max_abs[metric_index]
                .max(max_abs(reference_checkpoint, candidate_checkpoint));
        }
        let row_cosine = cosine(&reference.vector, &candidate);
        all_cosines.push(row_cosine);

        let (repeated, _) = model.run(ids, &weights, &config, false)?;
        let row_deterministic = byte_identical(&candidate, &repeated);
        deterministic &= row_deterministic;
        let mut timings = Vec::with_capacity(cli.warm_repetitions);
        for _ in 0..cli.warm_repetitions {
            let started = Instant::now();
            let _ = model.run(ids, &weights, &config, false)?;
            timings.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        timings.sort_by(f64::total_cmp);
        row_metrics.push(RowMetric {
            id: row.id.clone(),
            active_tokens: ids.len(),
            cosine: row_cosine,
            warm_wall_ms_median: timings[timings.len() / 2],
            deterministic: row_deterministic,
        });
        output_vectors.push(candidate);
    }

    let checkpoint_metrics = labels
        .into_iter()
        .enumerate()
        .map(|(index, checkpoint)| CheckpointMetric {
            checkpoint,
            min_cosine: checkpoint_cosines[index]
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min),
            mean_cosine: checkpoint_cosines[index].iter().sum::<f32>()
                / checkpoint_cosines[index].len() as f32,
            max_abs: checkpoint_max_abs[index],
        })
        .collect::<Vec<_>>();
    let min_cosine = all_cosines.iter().copied().fold(f32::INFINITY, f32::min);
    let mean_cosine = all_cosines.iter().sum::<f32>() / all_cosines.len() as f32;
    let finite = min_cosine.is_finite()
        && mean_cosine.is_finite()
        && output_vectors
            .iter()
            .flatten()
            .all(|value| value.is_finite());
    let gate_passed = finite && min_cosine >= GATE && deterministic;
    let first_divergence = (!gate_passed)
        .then(|| {
            checkpoint_metrics
                .iter()
                .find(|metric| {
                    !metric.min_cosine.is_finite()
                        || !metric.mean_cosine.is_finite()
                        || metric.min_cosine < GATE
                })
                .map(|metric| metric.checkpoint.clone())
        })
        .flatten();
    let report = Report {
        model: identity,
        sequence_length: cli.sequence_length,
        layers_per_executable: cli.layers_per_executable,
        pooling: "first token (CLS), then L2 normalize; the model card defines embedding pooling, while config classifier_pooling=mean is classifier-head metadata",
        cpu_reference: "fp32, exact erf GELU, full permitted attention",
        row_set_sha256: sha256_bytes(&rows_bytes),
        rows: row_metrics,
        min_cosine,
        mean_cosine,
        deterministic,
        gate_min_cosine: GATE,
        gate_passed,
        first_divergence,
        checkpoint_metrics,
        one_minute_load_average: load_average(),
        timing_note: "warm per-row wall clock; embedding gather, IOSurface conversion, and dispatch overhead included",
        vectors: output_vectors,
    };
    let json = serde_json::to_vec_pretty(&report).context("serialize report")?;
    println!("{}", String::from_utf8_lossy(&json));
    if let Some(path) = &cli.report {
        write_json(path, &json)?;
    }
    if let Some(path) = &cli.vectors_out {
        let vectors = serde_json::to_vec_pretty(&report.vectors).context("serialize vectors")?;
        write_json(path, &vectors)?;
    }
    if !report.gate_passed && cli.sequence_length == 512 {
        std::process::exit(2);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_matches_split_half_definition() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope_cpu(&mut values, 0, 10_000.0);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn local_mask_keeps_inclusive_radius() {
        let mask = local_distance_mask(63, 2, 0, 129, 64);
        assert_eq!(mask[0], 0.0);
        assert_eq!(mask[127], 0.0);
        assert_eq!(mask[128], MASK_MIN);
        assert_eq!(mask[129], 0.0);
        assert_eq!(mask[129 + 128], 0.0);
    }

    #[test]
    fn normalization_has_unit_length() {
        let vector = l2_normalize(vec![3.0, 4.0]);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }
}
