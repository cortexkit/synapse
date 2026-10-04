use crate::{modernbert, qwen};
use ane::{Executable, Graph, NSQualityOfService, TensorData};
use anyhow::{ensure, Context, Result};
use half::f16;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use synapse_core::{AneExecutable, AnePlacementInventory};

pub const REVISION: &str = "ane-direct-graph-v1";
pub const LADDER: [usize; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];
pub fn rung(tokens: usize) -> Result<usize> {
    ensure!(tokens > 0, "invalid_request");
    LADDER
        .into_iter()
        .find(|&n| n >= tokens)
        .context("context_limit_exceeded")
}

#[derive(Clone)]
pub struct Profile {
    pub id: String,
    pub model: Value,
    pub numeric: Value,
}
impl Profile {
    pub fn select(id: &str, operation: &str) -> Result<Self> {
        let manifest: Value =
            serde_json::from_slice(include_bytes!(concat!(env!("OUT_DIR"), "/models.json")))?;
        let numeric = manifest["profiles"][id].clone();
        ensure!(numeric["lane"] == "ane-direct-worker", "model_unsupported");
        let model =
            manifest["models"][numeric["model"].as_str().context("model_unsupported")?].clone();
        ensure!(model["operation"] == operation, "operation_mismatch");
        ensure!(numeric["compute_dtype"] == "f16", "model_unsupported");
        Ok(Self {
            id: id.into(),
            model,
            numeric,
        })
    }
    pub fn params(&self) -> &Value {
        &self.model["architecture"]["params"]
    }
    pub fn n(&self, key: &str) -> usize {
        self.params()[key].as_u64().expect("manifest integer") as usize
    }
    pub fn f(&self, key: &str) -> f32 {
        self.params()[key].as_f64().expect("manifest float") as f32
    }
    pub fn modern(&self) -> bool {
        self.model["architecture"]["family"] == "modernbert"
    }
    pub fn operation(&self) -> &str {
        self.model["operation"].as_str().unwrap()
    }
    pub fn prefix(&self) -> &str {
        self.model["tensor_prefix"].as_str().unwrap()
    }
    pub fn inventory(&self, shape: usize, os: &str) -> Result<AnePlacementInventory> {
        let stages: Vec<String> = serde_json::from_value(self.numeric["cpu_stages"].clone())?;
        let executables = (0..self.n("num_hidden_layers"))
            .map(|layer| AneExecutable {
                id: self.key(shape, layer, os),
                layers: vec![layer as u32],
            })
            .collect();
        let inventory = AnePlacementInventory {
            executables,
            cpu_stages: stages,
        };
        validate_inventory(&inventory, self.n("num_hidden_layers"))?;
        Ok(inventory)
    }
    pub fn key(&self, shape: usize, layer: usize, os: &str) -> String {
        let identity = serde_json::json!({"model_digest":self.numeric["converted_package_digest"],
            "operation": self.operation(), "numeric_profile":self.numeric,
            "rotation":self.numeric["rotation"], "shape":shape, "layer":layer,
            "graph_revision":REVISION, "worker_version":env!("CARGO_PKG_VERSION"), "os_build":os});
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&identity).unwrap())
        )
    }
}
pub fn validate_inventory(inventory: &AnePlacementInventory, layers: usize) -> Result<()> {
    const CPU: [&str; 8] = [
        "token_embedding",
        "mask_position",
        "rotation_in",
        "rotation_out",
        "final_norm",
        "pooling",
        "gte_classifier_head",
        "qwen_yes_no_readout",
    ];
    ensure!(
        inventory
            .cpu_stages
            .iter()
            .all(|stage| CPU.contains(&stage.as_str())),
        "invalid_placement"
    );
    let mut coverage = vec![0usize; layers];
    for executable in &inventory.executables {
        ensure!(!executable.layers.is_empty(), "invalid_placement");
        for &layer in &executable.layers {
            let count = coverage
                .get_mut(layer as usize)
                .context("invalid_placement")?;
            *count += 1;
        }
    }
    ensure!(coverage.into_iter().all(|n| n == 1), "invalid_placement");
    Ok(())
}

pub struct Model {
    pub profile: Profile,
    pub tensors: BTreeMap<String, Vec<f32>>,
    pub resident: BTreeMap<usize, Resident>,
}
pub struct Resident {
    pub inventory: AnePlacementInventory,
    executables: Vec<Executable>,
    a: TensorData,
    b: TensorData,
    residual: TensorData,
    mask: TensorData,
}
impl Model {
    pub fn load(profile: Profile, path: &Path, digest: &str) -> Result<Self> {
        ensure!(
            profile.numeric["converted_package_digest"] == digest,
            "package_digest_mismatch"
        );
        let path = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.into()
        };
        let bytes = std::fs::read(path)?;
        ensure!(
            format!("sha256:{:x}", Sha256::digest(&bytes)) == digest,
            "package_digest_mismatch"
        );
        let (metadata, st) = SafeTensors::read_metadata(&bytes)?;
        let _ = metadata;
        ensure!(
            st.metadata()
                .as_ref()
                .is_some_and(|m| m.get("profile") == Some(&profile.id)),
            "package_digest_mismatch"
        );
        let st = SafeTensors::deserialize(&bytes)?;
        if let Some(heads) = profile.model["head"]["tensors"].as_object() {
            for head in heads.values() {
                let key = head["key"].as_str().context("head_tensor_missing")?;
                let tensor = st
                    .tensor(key)
                    .map_err(|_| anyhow::anyhow!("head_tensor_missing"))?;
                ensure!(
                    serde_json::to_value(tensor.shape())? == head["shape"],
                    "head_tensor_missing"
                );
            }
        }
        if let Some(keys) = profile.model["head"]["forbidden_tensor_keys"].as_array() {
            ensure!(
                keys.iter().all(|k| st.tensor(k.as_str().unwrap()).is_err()),
                "head_tensor_missing"
            );
        }
        let fp32: Vec<String> = serde_json::from_value(profile.numeric["fp32_tensors"].clone())?;
        let mut tensors = BTreeMap::new();
        for (name, tensor) in st.tensors() {
            let expected = if fp32.contains(&name) {
                Dtype::F32
            } else {
                Dtype::F16
            };
            ensure!(tensor.dtype() == expected, "package_digest_mismatch");
            let values = match expected {
                Dtype::F32 => tensor
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                Dtype::F16 => tensor
                    .data()
                    .chunks_exact(2)
                    .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                    .collect(),
                _ => unreachable!(),
            };
            tensors.insert(name, values);
        }
        Ok(Self {
            profile,
            tensors,
            resident: BTreeMap::new(),
        })
    }
    pub fn tensor(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .get(name)
            .map(Vec::as_slice)
            .with_context(|| format!("tensor_missing:{name}"))
    }
    pub fn admit(&mut self, shape: usize, os: &str) -> Result<AnePlacementInventory> {
        ensure!(LADDER.contains(&shape), "invalid_shape");
        if let Some(resident) = self.resident.get(&shape) {
            return Ok(resident.inventory.clone());
        }
        ensure!(self.resident.len() < 4, "ane_residency_limit");
        let inventory = self.profile.inventory(shape, os)?;
        let hidden = self.profile.n("hidden_size");
        let mut executables = Vec::new();
        for layer in 0..self.profile.n("num_hidden_layers") {
            let mut graph = Graph::new();
            let input = graph.placeholder(modernbert::shape(shape, hidden));
            let mask = graph.placeholder(modernbert::shape(shape, 1));
            if self.profile.modern() {
                let residual = if layer == 0 {
                    graph.placeholder(modernbert::shape(shape, hidden))
                } else {
                    input
                };
                let config: modernbert::Config =
                    serde_json::from_value(self.profile.params().clone())?;
                let base = format!("{}layers.{layer}", self.profile.prefix());
                let linear = |name: &str| -> Result<modernbert::Linear> {
                    Ok(modernbert::Linear {
                        weight: self.tensor(&format!("{base}.{name}.weight"))?.to_vec(),
                    })
                };
                let weights = modernbert::LayerWeights {
                    qkv: linear("attn.Wqkv")?,
                    attention_output: linear("attn.Wo")?,
                    attention_norm: (layer > 0).then(|| vec![1.0; hidden]),
                    mlp_input: linear("mlp.Wi")?,
                    mlp_output: linear("mlp.Wo")?,
                    mlp_norm: vec![1.0; hidden],
                };
                let _ = modernbert::layer_graph(
                    &mut graph, input, residual, mask, &weights, &config, layer, shape,
                );
            } else {
                qwen::layer_graph(&mut graph, input, mask, self, layer, shape)?;
            }
            #[cfg(test)]
            let diagnostic = if std::env::var_os("ANE_DIAGNOSTICS").is_some() {
                Some(diagnostic_payload(&graph, shape, layer))
            } else {
                None
            };
            #[cfg(test)]
            let layer_started = std::time::Instant::now();
            let result = graph.compile(NSQualityOfService::UserInteractive);
            #[cfg(test)]
            if let Some(payload) = diagnostic {
                diagnostic_artifact(payload);
                println!("LAYER_LOAD shape={shape} layer={layer} prior_loaded={} elapsed_ms={:.3} outcome={}", executables.len(), layer_started.elapsed().as_secs_f64()*1000.0, result.as_ref().map(|_| "LOADED".to_owned()).unwrap_or_else(|e| e.to_string()));
            }
            executables.push(result.with_context(|| format!("compile layer {layer}"))?);
        }
        self.resident.insert(
            shape,
            Resident {
                inventory: inventory.clone(),
                executables,
                a: TensorData::new(modernbert::shape(shape, hidden)),
                b: TensorData::new(modernbert::shape(shape, hidden)),
                residual: TensorData::new(modernbert::shape(shape, hidden)),
                mask: TensorData::new(modernbert::shape(shape, 1)),
            },
        );
        Ok(inventory)
    }
    pub fn run(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.run_stages(tokens, false)
    }
    fn run_stages(&self, tokens: &[u32], traced: bool) -> Result<Vec<f32>> {
        let mut stages = StageClock::new(traced);
        let shape = rung(tokens.len())?;
        let resident = self.resident.get(&shape).context("shape_not_admitted")?;
        let hidden = self.profile.n("hidden_size");
        let pad = self.profile.n("pad_token_id") as u32;
        ensure!(!tokens.is_empty(), "invalid_request");
        let embedding_name = if self.profile.modern() {
            "embeddings.tok_embeddings.weight"
        } else {
            "embed_tokens.weight"
        };
        let embeddings = self.tensor(&format!("{}{embedding_name}", self.profile.prefix()))?;
        let mut input = vec![0.0; hidden * shape];
        for position in 0..shape {
            let token = tokens.get(position).copied().unwrap_or(pad) as usize;
            let embedding = embeddings
                .get(token * hidden..(token + 1) * hidden)
                .context("invalid_request")?;
            let mut row = embedding.to_vec();
            if self.profile.modern() {
                rms_cpu(&mut row, None, self.profile.f("norm_eps"));
            }
            for channel in 0..hidden {
                input[channel * shape + position] = row[channel];
            }
        }
        stages.emit("host_prologue");
        resident.a.copy_from_f32(&input);
        stages.emit("initial_input_copy_fp32_to_fp16");
        if self.profile.modern() {
            let matrix = self.tensor("rotation_in.weight")?;
            let mut residual = vec![0.0; input.len()];
            for position in 0..shape {
                let row: Vec<_> = (0..hidden).map(|c| input[c * shape + position]).collect();
                let mut row: Vec<f32> = (0..hidden)
                    .map(|c| {
                        row.iter()
                            .enumerate()
                            .map(|(i, v)| v * matrix[i * hidden + c])
                            .sum()
                    })
                    .collect();
                let direction = rotation_center_direction();
                let mean =
                    row.iter().zip(direction).map(|(v, q)| v * q).sum::<f32>() / hidden as f32;
                for c in 0..hidden {
                    row[c] -= mean * direction[c];
                    residual[c * shape + position] = row[c];
                }
            }
            stages.emit("rotation_in_fp32");
            resident.residual.copy_from_f32(&residual);
            stages.emit("residual_copy_fp32_to_fp16");
        }
        resident
            .mask
            .copy_from_f32(&padding_mask(tokens, pad, shape));
        stages.emit("mask_copy_fp32_to_fp16");
        for (layer, executable) in resident.executables.iter().enumerate() {
            let (src, dst) = if layer % 2 == 0 {
                (&resident.a, &resident.b)
            } else {
                (&resident.b, &resident.a)
            };
            let inputs: &[&TensorData] = if layer == 0 && self.profile.modern() {
                &[src, &resident.mask, &resident.residual]
            } else {
                &[src, &resident.mask]
            };
            if traced {
                let (prepare, evaluate, created) =
                    executable.run_cached_profiled(inputs, &[dst])?;
                println!("FORWARD_LAYER layer={layer} request_prepare_ms={:.6} sync_submit_and_wait_ms={:.6} request_created={created} interlayer_host_copy_bytes=0 iosurface_allocations=0", prepare.as_secs_f64()*1000.0, evaluate.as_secs_f64()*1000.0);
                stages.restart();
            } else {
                executable.run_cached(inputs, &[dst])?;
            }
        }
        let surface = if resident.executables.len() % 2 == 0 {
            &resident.a
        } else {
            &resident.b
        };
        let raw = surface.read_f32();
        stages.emit("final_readback_fp16_to_fp32");
        let mut rows = Vec::new();
        for (position, &token) in tokens.iter().enumerate() {
            let _ = token;
            let mut row: Vec<_> = (0..hidden).map(|c| raw[c * shape + position]).collect();
            if self.profile.modern() {
                rms_cpu(&mut row, None, self.profile.f("norm_eps"));
                row = linear_cpu(&row, self.tensor("rotation_out.weight")?, hidden)?;
            } else {
                rms_cpu(
                    &mut row,
                    Some(self.tensor(&format!("{}norm.weight", self.profile.prefix()))?),
                    self.profile.f("norm_eps"),
                );
            }
            rows.push(row);
        }
        stages.emit("rotation_out_and_final_norm_fp32");
        if self.profile.operation() == "embed" {
            let mut vector = if self.profile.modern() {
                rows.remove(0)
            } else {
                rows.pop().unwrap()
            };
            let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for value in &mut vector {
                *value /= norm;
            }
            stages.emit("cpu_head");
            return Ok(vector);
        }
        let score = if self.profile.modern() {
            let mut mean = vec![0.0; hidden];
            for row in &rows {
                for c in 0..hidden {
                    mean[c] += row[c] / rows.len() as f32;
                }
            }
            let mut dense = linear_cpu(&mean, self.tensor("head.dense.weight")?, hidden)?;
            for v in &mut dense {
                *v = 0.5 * *v * (1.0 + libm::erff(*v / std::f32::consts::SQRT_2));
            }
            let mean = dense.iter().sum::<f32>() / hidden as f32;
            for v in &mut dense {
                *v -= mean;
            }
            rms_cpu(
                &mut dense,
                Some(self.tensor("head.norm.weight")?),
                self.profile.f("norm_eps"),
            );
            let logit = linear_cpu(&dense, self.tensor("classifier.weight")?, 1)?[0]
                + self.tensor("classifier.bias")?[0];
            1.0 / (1.0 + (-logit).exp())
        } else {
            let row = rows.pop().unwrap();
            let readout = &self.profile.model["grammar"]["readout"];
            let yes = readout["yes"]["id"]
                .as_u64()
                .context("head_tensor_missing")? as usize;
            let no = readout["no"]["id"]
                .as_u64()
                .context("head_tensor_missing")? as usize;
            let dot = |token: usize| -> f32 {
                row.iter()
                    .zip(&embeddings[token * hidden..(token + 1) * hidden])
                    .map(|(a, b)| a * b)
                    .sum()
            };
            1.0 / (1.0 + (dot(no) - dot(yes)).exp())
        };
        stages.emit("cpu_head");
        Ok(vec![score])
    }
}
pub fn padding_mask(tokens: &[u32], pad: u32, shape: usize) -> Vec<f32> {
    let _ = pad;
    (0..shape)
        .map(|i| if i < tokens.len() { 0.0 } else { -10_000.0 })
        .collect()
}
pub fn linear_cpu(input: &[f32], matrix: &[f32], rows: usize) -> Result<Vec<f32>> {
    ensure!(matrix.len() == rows * input.len(), "tensor_shape_mismatch");
    Ok(matrix
        .chunks_exact(input.len())
        .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
        .collect())
}
pub fn rms_cpu(row: &mut [f32], weight: Option<&[f32]>, eps: f32) {
    let inv = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + eps)
        .sqrt()
        .recip();
    for (c, v) in row.iter_mut().enumerate() {
        *v *= inv * weight.map_or(1.0, |w| w[c]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn placement_requires_exact_layer_cover_and_closed_cpu_set() {
        let p = Profile::select("gte-modernbert-base.ane-direct-worker", "embed").unwrap();
        let mut inventory = p.inventory(128, "test-os").unwrap();
        inventory.executables.pop();
        assert!(validate_inventory(&inventory, 22).is_err());
        let mut inventory = p.inventory(128, "test-os").unwrap();
        inventory.cpu_stages.push("attention".into());
        assert!(validate_inventory(&inventory, 22).is_err());
        let mut inventory = p.inventory(128, "test-os").unwrap();
        inventory.executables[0].layers.push(1);
        assert!(validate_inventory(&inventory, 22).is_err());
    }
    #[test]
    fn compiled_identity_separates_rotation_dtype_operation_shape_and_os() {
        let p = Profile::select("gte-modernbert-base.ane-direct-worker", "embed").unwrap();
        let original = p.key(128, 0, "os-a");
        let mut changed = p.clone();
        changed.numeric["rotation"] = Value::String("none".into());
        assert_ne!(original, changed.key(128, 0, "os-a"));
        changed = p.clone();
        changed.numeric["compute_dtype"] = Value::String("f32".into());
        assert_ne!(original, changed.key(128, 0, "os-a"));
        assert_ne!(original, p.key(256, 0, "os-a"));
        assert_ne!(original, p.key(128, 1, "os-a"));
        assert_ne!(original, p.key(128, 0, "os-b"));
    }
    #[test]
    fn smallest_rung_and_right_padding_mask() {
        for (length, expected) in [(1, 128), (128, 128), (129, 256), (257, 512), (8192, 8192)] {
            assert_eq!(rung(length).unwrap(), expected);
        }
        assert!(rung(8193).is_err());
        assert!(rung(0).is_err());
        let mask = padding_mask(&[10, 11, 0, 12], 0, 128);
        assert_eq!(&mask[..5], &[0.0, 0.0, 0.0, 0.0, -10_000.0]);
        assert!(mask[4..].iter().all(|&v| v == -10_000.0));
    }
    #[test]
    fn load_does_not_compile_and_nonresident_inference_refuses() {
        let profile = Profile::select("gte-modernbert-base.ane-direct-worker", "embed").unwrap();
        let model = Model {
            profile,
            tensors: BTreeMap::new(),
            resident: BTreeMap::new(),
        };
        assert_eq!(
            model.run(&[50281, 50282]).unwrap_err().to_string(),
            "shape_not_admitted"
        );
    }
    #[test]
    #[ignore = "requires four converted packages in ANE_TEST_PACKAGES and private ANE hardware"]
    fn real_weight_short_parity_all_four_profiles() {
        crate::worker::test_private_api().unwrap();
        let root = std::path::PathBuf::from(
            std::env::var_os("ANE_TEST_PACKAGES").expect("ANE_TEST_PACKAGES"),
        );
        let mut failures = Vec::new();
        for slug in [
            "gte-modernbert-base",
            "gte-reranker-modernbert-base",
            "qwen3-embedding-0.6b",
            "qwen3-reranker-0.6b",
        ] {
            let operation = if slug.contains("reranker") {
                "rerank"
            } else {
                "embed"
            };
            let p = Profile::select(&format!("{slug}.ane-direct-worker"), operation).unwrap();
            let digest = p.numeric["converted_package_digest"]
                .as_str()
                .unwrap()
                .to_string();
            let mut model =
                Model::load(p, &root.join(format!("{slug}.safetensors")), &digest).unwrap();
            assert!(model.resident.is_empty());
            let fixture: Value = serde_json::from_slice(&std::fs::read(format!("../../bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json")).unwrap()).unwrap();
            let case = &fixture["cases"][0];
            let tokens: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
            let shape = rung(tokens.len()).unwrap();
            let start = std::time::Instant::now();
            let inventory = model.admit(shape, "hardware-test").unwrap();
            assert_eq!(inventory, model.admit(shape, "hardware-test").unwrap());
            let warm = std::time::Instant::now();
            let output = model.run(&tokens).unwrap();
            let warm_ms = warm.elapsed().as_secs_f64() * 1000.0;
            if operation == "embed" {
                let expected: Vec<f32> = serde_json::from_value(case["output"].clone()).unwrap();
                let cosine = output
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum::<f64>()
                    / (output.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                        * expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>())
                    .sqrt();
                println!(
                    "{slug}: shape={shape} cosine={cosine:.9} warm_ms={warm_ms:.3} total_ms={:.3}",
                    start.elapsed().as_secs_f64() * 1000.0
                );
                if cosine < 0.999 {
                    failures.push(format!("{slug} cosine={cosine}"));
                }
            } else {
                let expected = case["output"].as_f64().unwrap();
                let abs = (output[0] as f64 - expected).abs();
                println!("{slug}: shape={shape} score={} expected={expected} abs={abs:.9} warm_ms={warm_ms:.3} total_ms={:.3}", output[0], start.elapsed().as_secs_f64()*1000.0);
                if abs > 0.01 {
                    failures.push(format!("{slug} abs={abs}"));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("; "));
    }
}

fn rotation_center_direction() -> &'static Vec<f32> {
    static DIRECTION: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    DIRECTION.get_or_init(|| {
        // This vector is the manifest's generated Hadamard matrix transposed
        // and applied to an all-ones vector. Subtracting its projection removes
        // the original channel mean without undoing the residual rotation.
        let mut h = [[0i8; 12]; 12];
        for (i, row) in h.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let x = (j as i64 - i as i64).rem_euclid(11);
                let chi = if x == 0 {
                    0
                } else if (1..11).any(|v| v * v % 11 == x) {
                    1
                } else {
                    -1
                };
                let skew = match (i, j) {
                    (0, 0) => 0,
                    (0, _) => 1,
                    (_, 0) => -1,
                    _ => chi,
                };
                *cell = skew + i8::from(i == j);
            }
        }
        let mut matrix = Vec::new();
        let mut direction = vec![0.0f64; 768];
        let scale = 1.0 / 768f64.sqrt();
        for r in 0..768 {
            let mut hash = Sha256::new();
            hash.update(b"modernbert-hadamard-768");
            hash.update(0u64.to_le_bytes());
            hash.update(((r / 256) as u64).to_le_bytes());
            let digest = hash.finalize();
            let bit = r % 256;
            let sign = if (digest[bit / 8] >> (bit % 8)) & 1 == 1 {
                -1
            } else {
                1
            };
            for c in 0..768 {
                let sylvester = if ((r % 64) & (c % 64)).count_ones() % 2 == 0 {
                    1
                } else {
                    -1
                };
                let value = (sign * h[r / 64][c / 64] * sylvester) as f64 * scale;
                matrix.extend_from_slice(&value.to_le_bytes());
                direction[c] += value;
            }
        }
        assert_eq!(
            format!("{:x}", Sha256::digest(matrix)),
            "5cd34f0d01ce33615f987bd62be88b8e595f5083a3c94b42291c053d5cf87d53"
        );
        direction.into_iter().map(|v| v as f32).collect()
    })
}

#[cfg(test)]
mod ladder_hardware {
    use super::*;
    #[test]
    #[ignore = "requires pinned converted packages and private ANE hardware"]
    fn real_weight_admission_all_ladder_shapes() {
        crate::worker::test_private_api().unwrap();
        let root = std::path::PathBuf::from(
            std::env::var_os("ANE_TEST_PACKAGES").expect("ANE_TEST_PACKAGES"),
        );
        let mut requests = 0;
        for slug in [
            "gte-modernbert-base",
            "gte-reranker-modernbert-base",
            "qwen3-embedding-0.6b",
            "qwen3-reranker-0.6b",
        ] {
            let operation = if slug.contains("reranker") {
                "rerank"
            } else {
                "embed"
            };
            let profile = Profile::select(&format!("{slug}.ane-direct-worker"), operation).unwrap();
            let digest = profile.numeric["converted_package_digest"]
                .as_str()
                .unwrap()
                .to_string();
            let mut model =
                Model::load(profile, &root.join(format!("{slug}.safetensors")), &digest).unwrap();
            for shape in LADDER {
                model.resident.clear();
                let started = std::time::Instant::now();
                let result = model.admit(shape, "hardware-test");
                println!(
                    "{slug}: ADMIT shape={shape} compile_ms={:.3} outcome={}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    result
                        .as_ref()
                        .map(|_| "ADMITTED".into())
                        .unwrap_or_else(|e| format!("{e:#}"))
                );
                result.unwrap();
                requests += 1;
            }
        }
        assert_eq!(requests, 28);
    }
}

#[cfg(test)]
mod fresh_process_hardware {
    use super::*;
    fn model(slug: &str) -> Model {
        crate::worker::test_private_api().unwrap();
        let root = std::path::PathBuf::from(
            std::env::var_os("ANE_TEST_PACKAGES").expect("ANE_TEST_PACKAGES"),
        );
        let operation = if slug.contains("reranker") {
            "rerank"
        } else {
            "embed"
        };
        let profile = Profile::select(&format!("{slug}.ane-direct-worker"), operation).unwrap();
        let digest = profile.numeric["converted_package_digest"]
            .as_str()
            .unwrap()
            .to_owned();
        Model::load(profile, &root.join(format!("{slug}.safetensors")), &digest).unwrap()
    }
    fn resident_surface_bytes(model: &Model) {
        for (&shape, resident) in &model.resident {
            let sizes = [
                resident.a.surface().allocationSize(),
                resident.b.surface().allocationSize(),
                resident.residual.surface().allocationSize(),
                resident.mask.surface().allocationSize(),
            ];
            println!("RESIDENT_IO shape={shape} iosurface_bytes={sizes:?} total_bytes={} fp32_scratch_bytes={}", sizes.iter().sum::<isize>(), (3 * model.profile.n("hidden_size") + 1) * shape * 4);
        }
    }
    #[test]
    #[ignore = "fresh-process experiment with real weights; ANE_TEST_SHAPE required"]
    fn fresh_process_gte_single_shape_admission() {
        let mut model = model("gte-modernbert-base");
        assert!(model.resident.is_empty());
        let shape: usize = std::env::var("ANE_TEST_SHAPE").unwrap().parse().unwrap();
        let started = std::time::Instant::now();
        let result = model.admit(shape, "fresh-process-release");
        println!(
            "FRESH_SHAPE shape={shape} prior_resident=[] elapsed_ms={:.3} resident={:?} outcome={}",
            started.elapsed().as_secs_f64() * 1000.0,
            model.resident.keys().collect::<Vec<_>>(),
            result
                .as_ref()
                .map(|_| "ADMITTED".to_owned())
                .unwrap_or_else(|e| format!("{e:#}"))
        );
        resident_surface_bytes(&model);
        let success = result.is_ok();
        drop(result);
        drop(model);
        cleanup_diagnostic_artifacts();
        assert!(success, "shape admission failed; see per-layer diagnostics");
    }
    #[test]
    #[ignore = "fresh-process single-layer diagnostic with real weights"]
    fn fresh_process_gte_single_layer_load() {
        let model = model("gte-modernbert-base");
        let shape: usize = std::env::var("ANE_TEST_SHAPE").unwrap().parse().unwrap();
        let layer: usize = std::env::var("ANE_TEST_LAYER").unwrap().parse().unwrap();
        let hidden = model.profile.n("hidden_size");
        let mut graph = Graph::new();
        let input = graph.placeholder(modernbert::shape(shape, hidden));
        let mask = graph.placeholder(modernbert::shape(shape, 1));
        let residual = if layer == 0 {
            graph.placeholder(modernbert::shape(shape, hidden))
        } else {
            input
        };
        let config: modernbert::Config =
            serde_json::from_value(model.profile.params().clone()).unwrap();
        let base = format!("{}layers.{layer}", model.profile.prefix());
        let linear = |name: &str| modernbert::Linear {
            weight: model
                .tensor(&format!("{base}.{name}.weight"))
                .unwrap()
                .to_vec(),
        };
        let weights = modernbert::LayerWeights {
            qkv: linear("attn.Wqkv"),
            attention_output: linear("attn.Wo"),
            attention_norm: (layer > 0).then(|| vec![1.0; hidden]),
            mlp_input: linear("mlp.Wi"),
            mlp_output: linear("mlp.Wo"),
            mlp_norm: vec![1.0; hidden],
        };
        let _ = modernbert::layer_graph(
            &mut graph, input, residual, mask, &weights, &config, layer, shape,
        );
        let payload = diagnostic_payload(&graph, shape, layer);
        let started = std::time::Instant::now();
        let result = graph.compile(NSQualityOfService::UserInteractive);
        println!(
            "FRESH_LAYER shape={shape} layer={layer} elapsed_ms={:.3} outcome={}",
            started.elapsed().as_secs_f64() * 1000.0,
            result
                .as_ref()
                .map(|_| "LOADED".to_owned())
                .unwrap_or_else(|e| e.to_string())
        );
        diagnostic_artifact(payload);
        let success = result.is_ok();
        drop(result);
        cleanup_diagnostic_artifacts();
        assert!(success, "single layer failed; see diagnostics");
    }
    #[test]
    #[ignore = "real-weight 512-token latency; ANE_TEST_MODEL required"]
    fn real_weight_512_warm_latency() {
        let slug = std::env::var("ANE_TEST_MODEL").unwrap();
        let mut model = model(&slug);
        let fixture: Value = serde_json::from_slice(
            &std::fs::read(format!(
                "../../bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| {
                case["input_ids"]
                    .as_array()
                    .is_some_and(|ids| ids.len() == 512)
            })
            .expect("512-token fixture");
        let tokens: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
        let started = std::time::Instant::now();
        model.admit(512, "latency-test").unwrap();
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = std::time::Instant::now();
        let output = model.run(&tokens).unwrap();
        let first_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut times = Vec::new();
        for _ in 0..5 {
            let started = std::time::Instant::now();
            let repeat = model.run(&tokens).unwrap();
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(output, repeat, "nondeterministic warm result");
        }
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        let metric = if model.profile.operation() == "embed" {
            let expected: Vec<f32> = serde_json::from_value(case["output"].clone()).unwrap();
            let dot = output
                .iter()
                .zip(&expected)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>();
            let norm = (output.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                * expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>())
            .sqrt();
            format!("cosine={:.9}", dot / norm)
        } else {
            format!(
                "score={} expected={} abs={:.9}",
                output[0],
                case["output"],
                (output[0] as f64 - case["output"].as_f64().unwrap()).abs()
            )
        };
        println!("LATENCY model={slug} tokens=512 debug_assertions={} compile_ms={compile_ms:.3} first_ms={first_ms:.3} warm_ms={times:?} median_ms={:.3} {metric}", cfg!(debug_assertions), sorted[2]);
        if std::env::var_os("ANE_TRACE_FORWARD").is_some() {
            model.run_stages(&tokens, true).unwrap();
        }
        drop(model);
        cleanup_diagnostic_artifacts();
    }
}

#[cfg(test)]
fn diagnostic_paths() -> &'static std::sync::Mutex<Vec<std::path::PathBuf>> {
    static PATHS: std::sync::OnceLock<std::sync::Mutex<Vec<std::path::PathBuf>>> =
        std::sync::OnceLock::new();
    PATHS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}
#[cfg(test)]
fn diagnostic_payload(graph: &Graph, shape: usize, layer: usize) -> (String, String, usize, usize) {
    let (mil, weights) = graph.source_payload();
    let prefix = format!("{:X}_", Sha256::digest(mil.as_bytes()));
    println!("LAYER_PAYLOAD shape={shape} layer={layer} attention={} mil_bytes={} weight_bytes={} total_bytes={}", if layer.is_multiple_of(3) { "global" } else { "local" }, mil.len(), weights.len(), mil.len()+weights.len());
    (
        prefix,
        format!("{:x}", Sha256::digest(&weights)),
        mil.len(),
        weights.len(),
    )
}
#[cfg(test)]
fn diagnostic_artifact(
    (prefix, weight_digest, mil_bytes, weight_bytes): (String, String, usize, usize),
) {
    let paths: Vec<_> = std::fs::read_dir(std::env::temp_dir())
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.unwrap();
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(&prefix))
                .then(|| entry.path())
                .filter(|path| {
                    std::fs::read(path.join("weights/weight.bin"))
                        .is_ok_and(|bytes| format!("{:x}", Sha256::digest(bytes)) == weight_digest)
                })
        })
        .collect();
    assert_eq!(
        paths.len(),
        1,
        "expected one artifact for this exact MIL/weights payload"
    );
    let path = paths.into_iter().next().unwrap();
    assert_eq!(
        std::fs::metadata(path.join("model.mil")).unwrap().len() as usize,
        mil_bytes
    );
    assert_eq!(
        std::fs::metadata(path.join("weights/weight.bin"))
            .unwrap()
            .len() as usize,
        weight_bytes
    );
    diagnostic_paths().lock().unwrap().push(path);
}
#[cfg(test)]
fn cleanup_diagnostic_artifacts() {
    for path in diagnostic_paths().lock().unwrap().drain(..) {
        match std::fs::remove_dir_all(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("remove diagnostic model artifact: {error}"),
        }
    }
}

struct StageClock {
    enabled: bool,
    last: std::time::Instant,
}
impl StageClock {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last: std::time::Instant::now(),
        }
    }
    fn emit(&mut self, stage: &str) {
        if self.enabled {
            println!(
                "FORWARD_STAGE name={stage} elapsed_ms={:.6}",
                self.last.elapsed().as_secs_f64() * 1000.0
            );
            self.restart();
        }
    }
    fn restart(&mut self) {
        self.last = std::time::Instant::now();
    }
}
