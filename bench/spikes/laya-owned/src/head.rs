use anyhow::{bail, Result};
use half::f16;
use safetensors::{Dtype, SafeTensors};
use serde_json::{json, Value};
use std::{collections::HashMap, path::Path};

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
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
}

fn mm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, transpose_b: bool) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let mut c = vec![0.; m * n];
    unsafe {
        cblas_sgemm(
            101,
            111,
            if transpose_b { 112 } else { 111 },
            m as i32,
            n as i32,
            k as i32,
            1.,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            if transpose_b { k } else { n } as i32,
            0.,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}

pub fn load(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut out = HashMap::new();
    for (name, t) in tensors.tensors() {
        if name.starts_with("encoder.") {
            continue;
        }
        let v = match t.dtype() {
            Dtype::F16 => t
                .data()
                .chunks_exact(2)
                .map(|x| f16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32())
                .collect(),
            Dtype::F32 => t
                .data()
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect(),
            other => bail!("unsupported head dtype {other:?}"),
        };
        out.insert(name, v);
    }
    Ok(out)
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in x.iter_mut() {
        *v = (*v - max).exp();
    }
    let sum: f32 = x.iter().sum();
    for v in x {
        *v /= sum;
    }
}
fn gelu(x: f32) -> f32 {
    0.5 * x * (1. + libm::erff(x / std::f32::consts::SQRT_2))
}

pub struct Head {
    w: HashMap<String, Vec<f32>>,
    pub config: Value,
    pub dim: usize,
}
impl Head {
    pub fn new(path: &Path, config: Value) -> Result<Self> {
        let w = load(path)?;
        let dim = w["type_emb.weight"].len() / 3;
        Ok(Self { w, config, dim })
    }
    fn linear(&self, x: &[f32], name: &str, input: usize) -> Vec<f32> {
        let w = &self.w[&format!("{name}.weight")];
        let b = &self.w[&format!("{name}.bias")];
        let mut y = mm(x, w, x.len() / input, b.len(), input, true);
        for row in y.chunks_exact_mut(b.len()) {
            for (v, b) in row.iter_mut().zip(b) {
                *v += b;
            }
        }
        y
    }
    fn norm(&self, x: &[f32], name: &str) -> Vec<f32> {
        let w = &self.w[&format!("{name}.weight")];
        let b = &self.w[&format!("{name}.bias")];
        let mut y = x.to_vec();
        for row in y.chunks_exact_mut(w.len()) {
            let mean = row.iter().sum::<f32>() / w.len() as f32;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / w.len() as f32;
            for ((v, w), b) in row.iter_mut().zip(w).zip(b) {
                *v = (*v - mean) / (var + 1e-5).sqrt() * w + b;
            }
        }
        y
    }
    pub fn forward(
        &self,
        encoder: &[f32],
        mask: &[u8],
        markers: &[usize],
        qt: usize,
    ) -> (Value, Vec<f32>) {
        let d = self.dim;
        let seq = mask.len();
        assert_eq!(encoder.len(), seq * d);
        let mut h = encoder.to_vec();
        for row in h.chunks_exact_mut(d) {
            for (v, t) in row
                .iter_mut()
                .zip(&self.w["type_emb.weight"][qt * d..(qt + 1) * d])
            {
                *v += t;
            }
        }
        for layer in 0..2 {
            let prefix = format!("head.layers.{layer}");
            let norm = self.norm(&h, &format!("{prefix}.norm1"));
            let mut qkv = mm(
                &norm,
                &self.w[&format!("{prefix}.self_attn.in_proj_weight")],
                seq,
                3 * d,
                d,
                true,
            );
            for row in qkv.chunks_exact_mut(3 * d) {
                for (v, b) in row
                    .iter_mut()
                    .zip(&self.w[&format!("{prefix}.self_attn.in_proj_bias")])
                {
                    *v += b;
                }
            }
            let mut context = vec![0.; seq * d];
            let dh = d / 16;
            for head in 0..16 {
                let mut q = Vec::with_capacity(seq * dh);
                let mut k = Vec::with_capacity(seq * dh);
                let mut v = Vec::with_capacity(seq * dh);
                for row in qkv.chunks_exact(3 * d) {
                    q.extend_from_slice(&row[head * dh..(head + 1) * dh]);
                    k.extend_from_slice(&row[d + head * dh..d + (head + 1) * dh]);
                    v.extend_from_slice(&row[2 * d + head * dh..2 * d + (head + 1) * dh]);
                }
                let mut scores = mm(&q, &k, seq, seq, dh, true);
                for row in scores.chunks_exact_mut(seq) {
                    for (s, &m) in row.iter_mut().zip(mask) {
                        *s = if m == 0 {
                            f32::NEG_INFINITY
                        } else {
                            *s / (dh as f32).sqrt()
                        };
                    }
                    softmax(row);
                }
                let values = mm(&scores, &v, seq, dh, seq, false);
                for (dst, src) in context.chunks_exact_mut(d).zip(values.chunks_exact(dh)) {
                    dst[head * dh..(head + 1) * dh].copy_from_slice(src);
                }
            }
            let out = self.linear(&context, &format!("{prefix}.self_attn.out_proj"), d);
            for (x, y) in h.iter_mut().zip(out) {
                *x += y;
            }
            let n = self.norm(&h, &format!("{prefix}.norm2"));
            let mut ff = self.linear(&n, &format!("{prefix}.linear1"), d);
            for v in &mut ff {
                *v = v.max(0.);
            }
            let ff = self.linear(&ff, &format!("{prefix}.linear2"), 4 * d);
            for (x, y) in h.iter_mut().zip(ff) {
                *x += y;
            }
        }
        let gathered: Vec<f32> = markers
            .iter()
            .flat_map(|&m| h[m * d..(m + 1) * d].iter().copied())
            .collect();
        let mut z = self.linear(&self.norm(&gathered, "scorer.0"), "scorer.1", d);
        for v in &mut z {
            *v = gelu(*v);
        }
        let raw = self.linear(&z, "scorer.3", d);
        // Slots outside this row's option list are equivalent to -1e4 logits:
        // their f32 softmax contribution is zero, so they need not be materialized.
        let mut p = raw.clone();
        softmax(&mut p);
        let k = markers.len();
        let entropy = -p.iter().map(|v| v * v.max(1e-9).ln()).sum::<f32>() / (k.max(2) as f32).ln();
        let mut sorted = p.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let mut features = h[..d].to_vec();
        features.extend([
            sorted[0],
            sorted[0] - sorted[1],
            entropy,
            k.max(2) as f32 / 255.,
        ]);
        let mut act = self.linear(&features, "act_head.0", d + 4);
        for v in &mut act {
            *v = gelu(*v);
        }
        let mut act = self.linear(&act, "act_head.2", 256);
        let act_logits = act.clone();
        softmax(&mut act);
        let kind = ["choice", "score", "noul"][qt];
        let bucket = if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        };
        let temperature = self.config["temperature_by_options"][format!("{kind}:{bucket}")]
            .as_f64()
            .unwrap_or_else(|| self.config["temperature"][qt].as_f64().unwrap())
            as f32;
        let mut probs: Vec<f32> = raw.iter().map(|z| z / temperature.max(1e-3)).collect();
        softmax(&mut probs);
        let confidence = if qt == 2 {
            probs[1].max(1. - probs[1])
        } else if k < 2 {
            1.
        } else {
            (1. + probs
                .iter()
                .map(|p| p * p.clamp(1e-12, 1.).ln())
                .sum::<f32>()
                / (k as f32).ln())
            .clamp(0., 1.)
        };
        let score = if qt == 1 {
            Some(
                probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f32 * p)
                    .sum::<f32>(),
            )
        } else {
            None
        };
        (
            json!({"logits":raw,"temperature":temperature,"probabilities":probs,"confidence":confidence,"act_probability":act[0],"act_logits":act_logits,"score":score}),
            h,
        )
    }
}
