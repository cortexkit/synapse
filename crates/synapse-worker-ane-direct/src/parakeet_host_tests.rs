//! Recorded-encoder tests run CPU math only, without admitting any ANE programs.
use super::*;
use half::f16;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

// Checked against every logit, not just winning heads. Scalar bias-first folds
// and BLAS reorder f32 sums; the bound does not excuse any decision difference.
const ROUNDING_BOUND: f32 = 0.001;

struct Capture<'a> {
    joint: TdtJoint<'a>,
    logits: Vec<Vec<f32>>,
}

impl Joint for Capture<'_> {
    fn reset(&mut self) -> synapse_speech::Result<()> {
        self.joint.reset()
    }
    fn accept(&mut self, token: u32) -> synapse_speech::Result<()> {
        self.joint.accept(token)
    }
    fn evaluate(&mut self, frame: usize) -> synapse_speech::Result<JointLogits<'_>> {
        let logits = self.joint.evaluate(frame)?;
        self.logits.push(
            logits
                .tokens
                .iter()
                .chain(logits.durations)
                .copied()
                .collect(),
        );
        Ok(logits)
    }
}

struct Compare<'a> {
    joint: TdtJoint<'a>,
    expected: Vec<Vec<f32>>,
    step: usize,
    max_delta: f32,
}

impl Joint for Compare<'_> {
    fn reset(&mut self) -> synapse_speech::Result<()> {
        self.joint.reset()
    }
    fn accept(&mut self, token: u32) -> synapse_speech::Result<()> {
        self.joint.accept(token)
    }
    fn evaluate(&mut self, frame: usize) -> synapse_speech::Result<JointLogits<'_>> {
        let logits = self.joint.evaluate(frame)?;
        let expected = &self.expected[self.step];
        assert_eq!(expected.len(), logits.tokens.len() + logits.durations.len());
        for (&a, &b) in logits.tokens.iter().chain(logits.durations).zip(expected) {
            assert!(a.is_finite() && b.is_finite());
            let delta = (a - b).abs();
            self.max_delta = self.max_delta.max(delta);
            assert!(
                delta <= ROUNDING_BOUND,
                "step={} frame={frame} delta={delta} a={a} b={b}",
                self.step
            );
        }
        self.step += 1;
        Ok(logits)
    }
}

#[test]
fn accelerated_affine_matches_independent_f64_rows_and_rejects_bad_shapes() {
    let weight = [1.0, -2.0, 3.0, 0.5, 4.0, -8.0];
    let x = [0.25, 2.0, -0.5];
    let bias = [0.125, -0.25];
    let mut actual = bias;
    matrix_vector(&weight, &x, &mut actual).unwrap();
    for (row, got) in actual.into_iter().enumerate() {
        let expected = weight[row * 3..(row + 1) * 3]
            .iter()
            .zip(x)
            .fold(f64::from(bias[row]), |sum, (&w, x)| {
                sum + f64::from(w) * f64::from(x)
            });
        assert_eq!(f64::from(got), expected);
    }
    assert!(matrix_vector(&weight[..5], &x, &mut actual).is_err());
    assert!(matrix_vector(&[], &[], &mut []).is_err());
}

#[test]
fn host_timing_does_not_change_decoded_bytes() {
    let mut model = tests::synthetic_model(false);
    model.tensors.get_mut("joint.head.bias").unwrap()[3] = 10.0;
    model.tensors.get_mut("joint.head.bias").unwrap()[synapse_speech::VOCAB_SIZE + 4] = 10.0;
    let projected = vec![0.0; DECODER * 5];
    let mut joint = TdtJoint::new(&model, &projected, 5);
    joint.timing.enabled = false;
    assert!(joint.timing.start().is_none());
    let off = synapse_speech::greedy(&mut joint, 5, true, |_| {}).unwrap();
    let mut joint = TdtJoint::new(&model, &projected, 5);
    joint.timing.enabled = true;
    let on = synapse_speech::greedy(&mut joint, 5, true, |_| {}).unwrap();
    assert_eq!(
        serde_json::to_vec(&(off.triples, off.trace)).unwrap(),
        serde_json::to_vec(&(on.triples, on.trace)).unwrap()
    );
    assert!(joint.timing.predictor > Default::default());
    assert!(joint.timing.joint > Default::default());
    assert_eq!(joint.timing.predictor_calls(), 3);
}

#[test]
fn packed_surface_rows_preserve_fp16_values_and_padding() {
    for (shape, frames) in [(128, 1), (128, 127), (128, 128), (256, 129)] {
        let channels = 7;
        let values: Vec<_> = (0..shape * channels)
            .map(|i| f16::from_bits((i % 0x7c00) as u16).to_f32())
            .collect();
        let surface = ane::TensorData::with_f32(&values, crate::modernbert::shape(shape, channels));
        let actual = crate::backend::transcribe_rows(&surface, shape, channels, frames);
        for f in 0..frames {
            for c in 0..channels {
                assert_eq!(
                    actual[f * channels + c].to_bits(),
                    values[c * shape + f].to_bits()
                );
            }
        }
    }
}

#[test]
#[ignore = "CPU IOSurface readback timing, no ANE execution"]
fn readback_packing_measurement() {
    for (shape, frames) in [(128, 128), (256, 129), (1024, 769), (8192, 6001)] {
        let encoder = ane::TensorData::with_f32(
            &vec![0.125; shape * HIDDEN],
            crate::modernbert::shape(shape, HIDDEN),
        );
        let projection = ane::TensorData::with_f32(
            &vec![-0.25; shape * DECODER],
            crate::modernbert::shape(shape, DECODER),
        );
        for observe in [false, true] {
            let start = Instant::now();
            let raw = encoder.read_f32();
            let projected = projection.read_f32();
            let rows = |channels: usize, data: &[f32]| -> Vec<f32> {
                (0..frames)
                    .flat_map(|t| (0..channels).map(move |c| data[c * shape + t]))
                    .collect()
            };
            let old = (
                rows(DECODER, &projected),
                if observe { rows(HIDDEN, &raw) } else { vec![] },
            );
            std::hint::black_box(&raw);
            let old_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let new = (
                crate::backend::transcribe_rows(&projection, shape, DECODER, frames),
                if observe {
                    crate::backend::transcribe_rows(&encoder, shape, HIDDEN, frames)
                } else {
                    vec![]
                },
            );
            let new_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(old, new);
            println!("HOST_PACK shape={shape} frames={frames} observe={observe} old_ms={old_ms} new_ms={new_ms}");
        }
    }
}

fn checkpoint_model() -> Model {
    let snapshot = PathBuf::from(std::env::var_os("ANE_PARAKEET_SNAPSHOT").expect("snapshot"));
    let bytes = std::fs::read(snapshot.join("model.safetensors")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11"
    );
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let mut tensors = BTreeMap::new();
    for (name, t) in st.tensors() {
        if !name.starts_with("decoder.")
            && !name.starts_with("joint.")
            && !name.starts_with("encoder_projector.")
        {
            continue;
        }
        assert_eq!(t.dtype(), Dtype::F32);
        let values = t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| {
                let v = f32::from_le_bytes(*b);
                // Encoder weights have the same fp16 storage as the ANE package.
                if name.starts_with("encoder_projector.") {
                    f16::from_f32(v).to_f32()
                } else {
                    v
                }
            })
            .collect();
        tensors.insert(name, values);
    }
    Model {
        profile: Profile::select("parakeet-tdt-0.6b-v3.ane-direct-worker", "transcribe").unwrap(),
        tensors,
        resident: BTreeMap::new(),
    }
}

fn projection(model: &Model, encoder: &[f32], frames: usize) -> Vec<f32> {
    assert_eq!(encoder.len(), frames * HIDDEN);
    let weight = model.tensor("encoder_projector.weight").unwrap();
    let bias = model.tensor("encoder_projector.bias").unwrap();
    assert_eq!(weight.len(), DECODER * HIDDEN);
    assert_eq!(bias.len(), DECODER);
    let mut out = vec![0.0; frames * DECODER];
    // Fixture preparation only: A[frames,1024] times W[640,1024]^T.
    unsafe {
        crate::backend::cblas_sgemm(
            101,
            111,
            112,
            frames.try_into().unwrap(),
            DECODER as i32,
            HIDDEN as i32,
            1.0,
            encoder.as_ptr(),
            HIDDEN as i32,
            weight.as_ptr(),
            HIDDEN as i32,
            0.0,
            out.as_mut_ptr(),
            DECODER as i32,
        );
    }
    for row in out.as_chunks_mut::<DECODER>().0 {
        for (v, b) in row.iter_mut().zip(bias) {
            // Model the projected fp16 surface once, identically for both decoders.
            *v = f16::from_f32(*v + b).to_f32();
        }
    }
    out
}

#[test]
#[ignore = "CPU-only; requires pinned checkpoint and sealed encoder mirror"]
fn recorded_encoder_host_parity() {
    let model = checkpoint_model();
    let root = synapse_parity::parity_dir();
    let mirror = PathBuf::from(std::env::var_os("ANE_PARAKEET_MIRROR").expect("mirror"));
    let manifest = synapse_parity::manifest::Manifest::load(&root.join("models.json")).unwrap();
    let fixtures = synapse_parity::speech::load_transcribe_fixtures(&root, &manifest).unwrap();
    let mut count = 0;
    for fixture in fixtures.iter().filter(|f| f.encoder.is_some()) {
        let raw = fixture
            .encoder
            .as_ref()
            .unwrap()
            .read(&root, &mirror)
            .unwrap();
        let encoder: Vec<_> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let frames = fixture.valid_encoder_frames as usize;
        let projected = projection(&model, &encoder, frames);
        let mut scalar = TdtJoint::new(&model, &projected, frames);
        scalar.scalar = true;
        let mut captured = Capture {
            joint: scalar,
            logits: vec![],
        };
        let old = synapse_speech::greedy(&mut captured, frames, true, |_| {}).unwrap();
        let mut joint = TdtJoint::new(&model, &projected, frames);
        joint.timing.enabled = true;
        let start = Instant::now();
        let output = synapse_speech::greedy(&mut joint, frames, true, |_| {}).unwrap();
        println!(
            "HOST_ACCELERATE {} {}",
            fixture.id,
            joint
                .timing
                .report(frames, Default::default(), start.elapsed())
        );
        assert_eq!(old.triples, output.triples, "{}", fixture.id);
        assert_eq!(old.trace.len(), output.trace.len(), "{}", fixture.id);
        let mut max_score_delta = 0.0f32;
        for (old, new) in old.trace.iter().zip(&output.trace) {
            assert_eq!(
                (
                    old.frame_index,
                    old.token_top1,
                    old.duration_top1,
                    old.emitted,
                    old.cap_hit
                ),
                (
                    new.frame_index,
                    new.token_top1,
                    new.duration_top1,
                    new.emitted,
                    new.cap_hit
                ),
                "{}",
                fixture.id
            );
            for (a, b) in [
                (old.token_logit1, new.token_logit1),
                (old.token_logit2, new.token_logit2),
                (old.duration_logit1, new.duration_logit1),
                (old.duration_logit2, new.duration_logit2),
            ] {
                max_score_delta = max_score_delta.max((a - b).abs());
            }
        }
        assert!(max_score_delta <= ROUNDING_BOUND);
        assert_eq!(
            synapse_speech::Tokenizer::pinned()
                .transcript(&old.triples, fixture.duration_ms as f64)
                .unwrap(),
            synapse_speech::Tokenizer::pinned()
                .transcript(&output.triples, fixture.duration_ms as f64)
                .unwrap(),
            "{}",
            fixture.id
        );
        let mut compared = Compare {
            joint: TdtJoint::new(&model, &projected, frames),
            expected: captured.logits,
            step: 0,
            max_delta: 0.0,
        };
        let checked = synapse_speech::greedy(&mut compared, frames, true, |_| {}).unwrap();
        assert_eq!(checked, output);
        assert_eq!(compared.step, compared.expected.len());
        println!(
            "HOST_PARITY {} max_logit_delta={} max_score_delta={max_score_delta}",
            fixture.id, compared.max_delta
        );
        count += 1;
    }
    assert_eq!(count, 79, "every sealed recorded encoder must be exercised");
}
