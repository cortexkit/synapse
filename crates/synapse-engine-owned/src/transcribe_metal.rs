use super::Progress;
use crate::runtime::{safetensor_files, tensors_from_safetensors_bytes, Tensor};
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
        let loaded = load_weights(root, path)?;
        let gpu = Gpu::new()?;
        let weights = upload_weights(loaded, |name, tensor| {
            let c_name = CString::new(name)?;
            let dims: Vec<u64> = tensor.shape.iter().map(|&n| n as u64).collect();
            gpu.check(unsafe {
                synapse_parakeet_weight(
                    gpu.0.as_ptr(),
                    c_name.as_ptr(),
                    tensor.data.as_ptr(),
                    dims.as_ptr(),
                    dims.len() as u64,
                )
            })
        })?;
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

struct MappedWeights(NonNull<c_void>);

impl MappedWeights {
    fn open(path: &Path) -> Result<Self> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())?;
        NonNull::new(unsafe { synapse_parakeet_map_weights(path.as_ptr()) })
            .map(Self)
            .with_context(|| {
                format!(
                    "map safetensors: {}",
                    unsafe { CStr::from_ptr(synapse_parakeet_error()) }.to_string_lossy()
                )
            })
    }

    fn bytes(&self) -> &[u8] {
        let mut len = 0;
        // This owner keeps the read-only mapping alive until Drop. Tensor
        // conversion copies bytes before the mapping is released.
        unsafe {
            let bytes = synapse_parakeet_mapped_bytes(self.0.as_ptr(), &mut len);
            std::slice::from_raw_parts(bytes, len)
        }
    }
}

impl Drop for MappedWeights {
    fn drop(&mut self) {
        unsafe { synapse_parakeet_unmap_weights(self.0.as_ptr()) }
    }
}

/// Reads the checkpoint through a read-only file mapping instead of a read
/// buffer. A 2.5 GB read buffer, the converted host tensors and the GPU copies
/// used to be alive at once. A read-only mapping is backed by the file itself,
/// so the OS can drop its pages instead of holding a second private copy, and
/// each mapping is released once its tensors are converted.
fn load_weights(root: &Path, original: &Path) -> Result<HashMap<String, Tensor>> {
    let mut host = HashMap::new();
    for file in safetensor_files(root, original)? {
        let mapping = MappedWeights::open(&file)?;
        host.extend(tensors_from_safetensors_bytes(&file, mapping.bytes())?);
    }
    ensure!(
        host.values().all(|t| t.data.iter().all(|x| x.is_finite())),
        "non-finite model weight"
    );
    Ok(host)
}

fn upload_weights(
    loaded: HashMap<String, Tensor>,
    mut upload: impl FnMut(&str, &Tensor) -> Result<()>,
) -> Result<HashMap<String, Tensor>> {
    let mut host = HashMap::new();
    // Preserve host conversion before Metal allocation, but release each
    // encoder tensor as soon as its synchronous upload has copied its bytes.
    for (name, tensor) in loaded {
        if name.starts_with("encoder.") || name.starts_with("encoder_projector.") {
            upload(&name, &tensor)?;
        } else {
            host.insert(name, tensor);
        }
    }
    Ok(host)
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
    fn synapse_parakeet_map_weights(path: *const c_char) -> *mut c_void;
    fn synapse_parakeet_mapped_bytes(raw: *mut c_void, len: *mut usize) -> *const u8;
    fn synapse_parakeet_unmap_weights(raw: *mut c_void);
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

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/scratch")
                .join(format!(
                    "parakeet-weights-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, file: &str, name: &str, dtype: Dtype, bytes: &[u8], count: usize) {
            let view = TensorView::new(dtype, vec![count], bytes).unwrap();
            serialize_to_file([(name, view)], None, &self.0.join(file)).unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn mapped_weights_keep_f32_bits_and_only_retain_host_tensors() {
        let fixture = Fixture::new();
        let bits = [0x8000_0000_u32, 1, 0x7f7f_ffff, 0xbf80_0000];
        let bytes: Vec<_> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        fixture.write(
            "encoder.safetensors",
            "encoder.weight",
            Dtype::F32,
            &bytes,
            4,
        );
        fixture.write("host.safetensors", "decoder.weight", Dtype::F32, &bytes, 4);
        std::fs::write(fixture.0.join("model.safetensors.index.json"),
            br#"{"weight_map":{"encoder.weight":"encoder.safetensors","decoder.weight":"host.safetensors"}}"#).unwrap();
        let mut uploads = 0;
        let loaded = load_weights(&fixture.0, &fixture.0).unwrap();
        let host = upload_weights(loaded, |name, tensor| {
            assert_eq!(name, "encoder.weight");
            assert_eq!(tensor.shape, [4]);
            assert_eq!(
                tensor.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                bits
            );
            uploads += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(uploads, 1);
        assert_eq!(host.len(), 1);
        assert_eq!(
            host["decoder.weight"]
                .data
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
    }

    #[test]
    fn mapped_weights_validate_file_length_before_upload() {
        let fixture = Fixture::new();
        fixture.write(
            "model.safetensors",
            "encoder.weight",
            Dtype::F32,
            &1_f32.to_le_bytes(),
            1,
        );
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(fixture.0.join("model.safetensors"))
            .unwrap()
            .write_all(&[0])
            .unwrap();
        let error = load_weights(&fixture.0, &fixture.0).unwrap_err();
        assert!(format!("{error:#}").contains("load safetensors"));
    }

    #[test]
    fn mapped_weights_preserve_half_conversion_and_reject_nonfinite() {
        let fixture = Fixture::new();
        for (dtype, bits) in [(Dtype::F16, 0x3c00_u16), (Dtype::BF16, 0x3f80)] {
            fixture.write(
                "model.safetensors",
                "encoder.weight",
                dtype,
                &bits.to_le_bytes(),
                1,
            );
            let loaded = load_weights(&fixture.0, &fixture.0).unwrap();
            assert_eq!(loaded["encoder.weight"].data[0].to_bits(), 1_f32.to_bits());
        }
        fixture.write(
            "model.safetensors",
            "encoder.weight",
            Dtype::F32,
            &f32::NAN.to_le_bytes(),
            1,
        );
        let error = load_weights(&fixture.0, &fixture.0).unwrap_err();
        assert!(error.to_string().contains("non-finite model weight"));
    }

    #[test]
    fn mapped_weights_report_mapping_errors() {
        let fixture = Fixture::new();
        let error = MappedWeights::open(&fixture.0.join("missing.safetensors"))
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains("map safetensors: open"));
    }
}
