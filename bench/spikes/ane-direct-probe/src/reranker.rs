//! CPU sequence-classification head beside the shared fp16 ANE graph builders.
use super::*;
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
struct Pair {
    id: String,
    input_ids: Vec<u32>,
    reference_pool: Option<Vec<f32>>,
}

struct Head {
    dense: Linear,
    norm: Vec<f32>,
    classifier: Linear,
    bias: f32,
    eps: f32,
}

impl Head {
    fn load(snapshot: &Path, config: &Config) -> Result<Self> {
        let raw: serde_json::Value =
            serde_json::from_slice(&fs::read(snapshot.join("config.json"))?)?;
        for (key, expected) in [
            ("hidden_size", serde_json::json!(768)),
            ("intermediate_size", serde_json::json!(1152)),
            ("num_hidden_layers", serde_json::json!(22)),
            ("num_attention_heads", serde_json::json!(12)),
            ("global_attn_every_n_layers", serde_json::json!(3)),
            ("local_attention", serde_json::json!(128)),
            ("global_rope_theta", serde_json::json!(160000.0)),
            ("local_rope_theta", serde_json::json!(10000.0)),
            ("classifier_pooling", serde_json::json!("mean")),
            ("classifier_activation", serde_json::json!("gelu")),
            ("hidden_activation", serde_json::json!("gelu")),
            ("attention_bias", serde_json::json!(false)),
            ("mlp_bias", serde_json::json!(false)),
            ("norm_bias", serde_json::json!(false)),
            ("classifier_bias", serde_json::json!(false)),
        ] {
            ensure!(
                raw[key] == expected,
                "unsupported config {key}: {}",
                raw[key]
            );
        }
        let bytes = fs::read(snapshot.join("model.safetensors"))?;
        let st = SafeTensors::deserialize(&bytes)?;
        let h = config.hidden_size;
        Ok(Self {
            dense: load_linear(&st, "head.dense", h, h)?,
            norm: load_vector(&st, "head.norm.weight", h)?,
            classifier: load_linear(&st, "classifier", 1, h)?,
            bias: load_vector(&st, "classifier.bias", 1)?[0],
            eps: config.norm_eps,
        })
    }

    fn score(&self, pooled: &[f32]) -> f32 {
        let h = self.norm.len();
        let mut dense = vec![0.0_f32; h];
        for (i, value) in dense.iter_mut().enumerate() {
            let x: f32 = self.dense.weight[i * h..(i + 1) * h]
                .iter()
                .zip(pooled)
                .map(|(a, b)| a * b)
                .sum();
            // Transformers' "gelu" uses erf, not the encoder graph's tanh approximation.
            *value = 0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2));
        }
        let mean = dense.iter().sum::<f32>() / h as f32;
        let variance = dense.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / h as f32;
        let inverse = (variance + self.eps).sqrt().recip();
        self.bias
            + dense
                .iter()
                .enumerate()
                .map(|(i, x)| (x - mean) * inverse * self.norm[i] * self.classifier.weight[i])
                .sum::<f32>()
    }
}

fn encode(model: &AneModel, ids: &[u32], weights: &Weights, config: &Config) -> Result<Vec<f32>> {
    let width = model.shape.width;
    model
        .raw
        .copy_from_f32(&raw_embeddings(ids, weights, config, width)?);
    let mut mask = vec![MASK_MIN; width];
    // The tokenizer's unpadded pair is fully attended, even if it contains a literal pad token.
    mask[..ids.len()].fill(0.0);
    model.key_mask.copy_from_f32(&mask);
    model
        .embedding
        .run_cached(&[&model.raw], &[&model.hidden_a])?;
    for (index, executable) in model.chunks.iter().enumerate() {
        let (source, destination) = if index % 2 == 0 {
            (&model.hidden_a, &model.hidden_b)
        } else {
            (&model.hidden_b, &model.hidden_a)
        };
        executable.run_cached(&[source, &model.key_mask], &[destination])?;
    }
    let source = if model.chunks.len().is_multiple_of(2) {
        &model.hidden_a
    } else {
        &model.hidden_b
    };
    model.final_norm.run_cached(&[source], &[&model.raw])?;
    Ok(model.raw.read_f32().into_vec())
}

pub fn main() -> Result<()> {
    let snapshot = PathBuf::from(
        std::env::args()
            .nth(1)
            .context("usage: modernbert_rerank SNAPSHOT < pairs.jsonl")?,
    );
    let (config, weights, identity) = load_model(&snapshot)?;
    let head = Head::load(&snapshot, &config)?;
    let mut models = BTreeMap::new();
    println!("{}", serde_json::json!({"ready":true,"model":identity}));
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let pair: Pair = serde_json::from_str(&line?)?;
        ensure!(!pair.input_ids.is_empty(), "empty pair");
        ensure!(
            pair.input_ids.len() <= config.max_position_embeddings,
            "pair exceeds model limit; truncation forbidden"
        );
        let width = pair.input_ids.len().div_ceil(64) * 64;
        ensure!(
            width <= config.max_position_embeddings,
            "padded shape exceeds model limit"
        );
        let mut compile_ms = None;
        if let std::collections::btree_map::Entry::Vacant(entry) = models.entry(width) {
            let start = Instant::now();
            entry.insert(compile_model(&config, &weights, width, 1)?);
            compile_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
        }
        let total = Instant::now();
        let start = Instant::now();
        let hidden = encode(&models[&width], &pair.input_ids, &weights, &config)?;
        let encoder_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let pooled: Vec<f32> = (0..config.hidden_size)
            .map(|c| {
                hidden[c * width..c * width + pair.input_ids.len()]
                    .iter()
                    .sum::<f32>()
                    / pair.input_ids.len() as f32
            })
            .collect();
        let logit = head.score(&pooled);
        let head_ms = start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total.elapsed().as_secs_f64() * 1000.0;
        ensure!(logit.is_finite(), "nonfinite logit");
        let reference_head_logit = match pair.reference_pool {
            Some(pool) => {
                ensure!(
                    pool.len() == config.hidden_size,
                    "invalid reference pool shape"
                );
                Some(head.score(&pool))
            }
            None => None,
        };
        println!(
            "{}",
            serde_json::json!({"id":pair.id,"tokens":pair.input_ids.len(),"shape":width,
            "logit":logit,"encoder_ms":encoder_ms,"cpu_head_ms":head_ms,"total_ms":total_ms,
            "compile_ms":compile_ms,"reference_head_logit":reference_head_logit,"pooled":pooled})
        );
        io::stdout().flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_head_matches_torch_fp32_dense_gelu_norm_classifier() {
        let head = Head {
            dense: Linear {
                weight: vec![1.0, 2.0, -1.0, 1.0],
                rows: 2,
                columns: 2,
            },
            norm: vec![0.75, 1.25],
            classifier: Linear {
                weight: vec![2.0, -3.0],
                rows: 1,
                columns: 2,
            },
            bias: 0.125,
            eps: 1e-5,
        };
        // Independent torch 2.14 fp32 fixtures, including zero variance and negative GELU input.
        for (pool, expected) in [
            ([0.5, -0.25], 5.3713694_f32),
            ([1.0, 2.0], 5.374994_f32),
            ([0.0, 0.0], 0.125_f32),
        ] {
            assert!((head.score(&pool) - expected).abs() < 2e-6);
        }
    }
}
