use super::Progress;
use crate::runtime::{load_safetensor_map, Tensor};
use anyhow::{bail, ensure, Context as _, Result};
use std::{
    collections::HashMap,
    ffi::{c_char, c_void, CStr, CString},
    path::Path,
    ptr::NonNull,
};
use synapse_core::{RuntimeConfig, TranscribeOutput, TranscribeRequest};
use synapse_speech::{Joint, JointLogits};

pub(super) struct Model {
    gpu: Gpu,
    weights: HashMap<String, Tensor>,
    pub maximum: usize,
}

impl Model {
    pub fn load(cfg: &RuntimeConfig) -> Result<Self> {
        for (key, expected) in [
            ("family", "parakeet-tdt"),
            ("dtype", "f32"),
            ("execution", "explicit"),
        ] {
            ensure!(
                cfg.values.get(key).is_none_or(|s| s == expected),
                "unsupported transcribe {key}"
            );
        }
        let maximum: usize = cfg
            .values
            .get("max_encoder_frames")
            .context("missing max_encoder_frames")?
            .parse()?;
        ensure!(
            (1..=super::CERTIFIED_MAX_ENCODER_FRAMES).contains(&maximum),
            "max_encoder_frames exceeds certified Metal envelope"
        );
        let path = Path::new(cfg.values.get("model_path").context("missing model_path")?);
        let root = if path.is_file() {
            path.parent().context("model root")?
        } else {
            path
        };
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        ensure!(
            config["model_type"] == "parakeet_tdt"
                && config["encoder_config"]["hidden_size"] == 1024
                && config["encoder_config"]["num_hidden_layers"] == 24
                && config["decoder_hidden_size"] == 640,
            "unsupported Parakeet architecture"
        );
        let mut weights = load_safetensor_map(root, path)?;
        ensure!(
            weights
                .values()
                .all(|t| t.data.iter().all(|x| x.is_finite())),
            "non-finite model weight"
        );
        let gpu = Gpu::new()?;
        for (name, tensor) in &weights {
            if name.starts_with("encoder.") || name.starts_with("encoder_projector.") {
                let c_name = CString::new(name.as_str())?;
                let dims: Vec<u64> = tensor.shape.iter().map(|&n| n as u64).collect();
                gpu.check(unsafe {
                    synapse_parakeet_weight(
                        gpu.0.as_ptr(),
                        c_name.as_ptr(),
                        tensor.data.as_ptr(),
                        dims.as_ptr(),
                        dims.len() as u64,
                    )
                })?;
            }
        }
        weights.retain(|name, _| {
            !name.starts_with("encoder.") && !name.starts_with("encoder_projector.")
        });
        // Validate every host matvec before passing its dimensions to BLAS.
        for (name, shape) in [
            ("decoder.embedding.weight", vec![8193, 640]),
            ("decoder.decoder_projector.weight", vec![640, 640]),
            ("decoder.decoder_projector.bias", vec![640]),
            ("joint.head.weight", vec![8198, 640]),
            ("joint.head.bias", vec![8198]),
        ] {
            ensure!(
                weights.get(name).is_some_and(|t| t.shape == shape),
                "bad {name}"
            );
        }
        for layer in 0..2 {
            for kind in ["ih", "hh"] {
                for (field, shape) in [("weight", vec![2560, 640]), ("bias", vec![2560])] {
                    let name = format!("decoder.lstm.{field}_{kind}_l{layer}");
                    ensure!(
                        weights.get(&name).is_some_and(|t| t.shape == shape),
                        "bad {name}"
                    );
                }
            }
        }
        Ok(Self {
            gpu,
            weights,
            maximum,
        })
    }

    pub fn run(
        &mut self,
        request: TranscribeRequest,
        valid: usize,
        rung: usize,
        progress: &mut impl FnMut(Progress),
    ) -> Result<TranscribeOutput> {
        let mut mel = vec![0.0; rung * 8 * 128];
        let count = mel.len().min(request.mel.len());
        mel[..count].copy_from_slice(&request.mel[..count]);
        self.gpu.check(unsafe {
            synapse_parakeet_begin(
                self.gpu.0.as_ptr(),
                mel.as_ptr(),
                rung as u64,
                request.valid_mel_frames as u64,
            )
        })?;
        for layer in 0..24 {
            self.gpu
                .check(unsafe { synapse_parakeet_layer(self.gpu.0.as_ptr(), layer) })?;
            progress(Progress::EncoderLayersCompleted(layer as usize + 1));
        }
        // Read the same results in both modes. Observation only retains a copy.
        let mut encoder = vec![0.0; rung * 1024];
        let mut projected = vec![0.0; rung * 640];
        self.gpu.check(unsafe {
            synapse_parakeet_finish(
                self.gpu.0.as_ptr(),
                encoder.as_mut_ptr(),
                projected.as_mut_ptr(),
            )
        })?;
        ensure!(
            encoder.iter().chain(&projected).all(|x| x.is_finite()),
            "non-finite encoder output"
        );
        projected.truncate(valid * 640);
        let mut joint = Predictor::new(&self.weights, projected);
        Ok(super::decode(
            &mut joint,
            valid,
            request.observe,
            encoder,
            progress,
        )?)
    }
}

struct Gpu(NonNull<c_void>);
// Each loaded model owns its context and is serialized by the engine's mutex.
unsafe impl Send for Gpu {}
impl Gpu {
    fn new() -> Result<Self> {
        NonNull::new(unsafe { synapse_parakeet_new() })
            .map(Self)
            .context("no Metal device")
    }
    fn check(&self, status: i32) -> Result<()> {
        if status != 0 {
            bail!(
                "Parakeet MPSGraph: {}",
                unsafe { CStr::from_ptr(synapse_parakeet_error()) }.to_string_lossy()
            );
        }
        Ok(())
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe { synapse_parakeet_free(self.0.as_ptr()) }
    }
}

struct Predictor<'a> {
    weights: &'a HashMap<String, Tensor>,
    encoder: Vec<f32>,
    h: [Vec<f32>; 2],
    c: [Vec<f32>; 2],
    g: Vec<f32>,
    logits: Vec<f32>,
}
impl<'a> Predictor<'a> {
    fn new(weights: &'a HashMap<String, Tensor>, encoder: Vec<f32>) -> Self {
        Self {
            weights,
            encoder,
            h: std::array::from_fn(|_| vec![0.0; 640]),
            c: std::array::from_fn(|_| vec![0.0; 640]),
            g: vec![0.0; 640],
            logits: vec![0.0; 8198],
        }
    }
    fn predict(&mut self, token: u32) {
        let embedding = &self.weights["decoder.embedding.weight"].data;
        let mut input = embedding[token as usize * 640..(token as usize + 1) * 640].to_vec();
        for layer in 0..2 {
            let p = "decoder.lstm.";
            let mut gates = self.weights[&format!("{p}bias_ih_l{layer}")].data.clone();
            matvec(
                &self.weights[&format!("{p}weight_ih_l{layer}")].data,
                &input,
                &mut gates,
            );
            let mut recurrent = self.weights[&format!("{p}bias_hh_l{layer}")].data.clone();
            matvec(
                &self.weights[&format!("{p}weight_hh_l{layer}")].data,
                &self.h[layer],
                &mut recurrent,
            );
            for (g, r) in gates.iter_mut().zip(recurrent) {
                *g += r;
            }
            let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
            for i in 0..640 {
                let cell = sigmoid(gates[i + 640]) * self.c[layer][i]
                    + sigmoid(gates[i]) * gates[i + 1280].tanh();
                self.c[layer][i] = cell;
                self.h[layer][i] = sigmoid(gates[i + 1920]) * cell.tanh();
            }
            input.clone_from(&self.h[layer]);
        }
        self.g
            .clone_from(&self.weights["decoder.decoder_projector.bias"].data);
        matvec(
            &self.weights["decoder.decoder_projector.weight"].data,
            &input,
            &mut self.g,
        );
    }
}
impl Joint for Predictor<'_> {
    fn reset(&mut self) -> synapse_speech::Result<()> {
        for v in self.h.iter_mut().chain(self.c.iter_mut()) {
            v.fill(0.0);
        }
        self.predict(synapse_speech::BLANK);
        Ok(())
    }
    fn evaluate(&mut self, frame: usize) -> synapse_speech::Result<JointLogits<'_>> {
        let row = self.encoder.get(frame * 640..(frame + 1) * 640).ok_or(
            synapse_speech::Error::InvalidJoint("frame past valid encoder output"),
        )?;
        let input: Vec<f32> = row
            .iter()
            .zip(&self.g)
            .map(|(a, b)| (a + b).max(0.0))
            .collect();
        self.logits
            .clone_from(&self.weights["joint.head.bias"].data);
        matvec(
            &self.weights["joint.head.weight"].data,
            &input,
            &mut self.logits,
        );
        Ok(JointLogits {
            tokens: &self.logits[..8193],
            durations: &self.logits[8193..],
        })
    }
    fn accept(&mut self, token: u32) -> synapse_speech::Result<()> {
        if token >= synapse_speech::BLANK {
            return Err(synapse_speech::Error::InvalidJoint(
                "invalid predictor token",
            ));
        }
        self.predict(token);
        Ok(())
    }
}

fn matvec(weight: &[f32], input: &[f32], output: &mut [f32]) {
    assert_eq!(weight.len(), input.len() * output.len());
    // BLAS row-major, no transpose. beta=1 preserves the linear layer's bias.
    unsafe {
        cblas_sgemv(
            101,
            111,
            output.len() as i32,
            input.len() as i32,
            1.0,
            weight.as_ptr(),
            input.len() as i32,
            input.as_ptr(),
            1,
            1.0,
            output.as_mut_ptr(),
            1,
        );
    }
}
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
    fn synapse_parakeet_new() -> *mut c_void;
    fn synapse_parakeet_free(raw: *mut c_void);
    fn synapse_parakeet_error() -> *const c_char;
    fn synapse_parakeet_weight(
        raw: *mut c_void,
        name: *const c_char,
        data: *const f32,
        dims: *const u64,
        rank: u64,
    ) -> i32;
    fn synapse_parakeet_begin(raw: *mut c_void, mel: *const f32, rung: u64, valid: u64) -> i32;
    fn synapse_parakeet_layer(raw: *mut c_void, layer: u64) -> i32;
    fn synapse_parakeet_finish(raw: *mut c_void, encoder: *mut f32, projected: *mut f32) -> i32;
}
