//! Conversion rule `v1`: pinned checkpoint to canonical converted package.
//!
//! Each worker profile (CUDA, Vulkan, direct ANE) loads exactly one package,
//! produced only by this converter. For every output tensor:
//!
//! - a tensor in the profile's fp32 list is stored as F32. An untransformed
//!   tensor is widened from the checkpoint exactly (f16, bf16 and f32 all
//!   widen to f32 without rounding), so an f32 source is copied bit for bit;
//! - every other tensor is stored as F16 with IEEE round-to-nearest-even,
//!   `half::f16::from_f32` applied to the value widened to f32. A finite value
//!   that rounds to infinity is an error, not a silent overflow;
//! - when the profile names a rotation, the rotated tensors are computed in
//!   f64 with a fixed summation order, rounded once to f32, and then stored
//!   by the two rules above.
//!
//! The package layout is `safetensors::write_package`: tensors in
//! lexicographic name order and `__metadata__` exactly the profile id and
//! `conversion_rule: v1`. Its SHA-256 is the profile's
//! `converted_package_digest`.
//!
//! ModernBERT residual-stream rotation (scope `modernbert-residual-stream-v1`,
//! after the reference construction measured in
//! docs/evidence/ane-modernbert-rotation-conditioning/README.md). Weights use
//! the checkpoint's `[out, in]` layout, applied as `y = x Wᵀ` to row vectors;
//! `Q` is the `d × d` matrix `hadamard::generate` builds, `C = I − 11ᵀ/d`, and `γ` are the
//! LayerNorm scales:
//!
//! - token embeddings: each row centered, then `E C Q`;
//! - each layer's attention QKV and MLP input weights: the consuming norm's
//!   `γ` folded into the input columns, then right-multiplied by `Q` (layer
//!   0's QKV takes the embedding norm's `γ`, since layer 0 has no attention
//!   norm);
//! - each layer's attention output and MLP output weights: centered over
//!   output rows, then left-multiplied by `Qᵀ`;
//! - new `rotation_in.weight` = `Qᵀ C diag(γ_embedding) Q`, the first layer's
//!   residual projection, and new `rotation_out.weight` = `diag(γ_final) Q`,
//!   the final unrotation;
//! - the four norm scale families are folded away and not stored, so every
//!   norm becomes parameter-free.

use std::collections::BTreeMap;
use std::path::Path;

use crate::arch::expected_tensors;
use crate::canonical::sha256_file;
use crate::hadamard;
use crate::manifest::{Family, Fp32Tensors, Manifest, Model, Profile};
use crate::rules::{MODERNBERT_ROTATION, ROTATION_IN_KEY, ROTATION_OUT_KEY};
use crate::safetensors::{write_package, Checkpoint, PackageTensor, StDType};
use crate::{perr, Result};

/// f16 rounding mode. Only `NearestEven` is conversion rule `v1`; the other
/// mode exists so tests can show a package rounded toward zero is caught.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rounding {
    NearestEven,
    #[cfg_attr(not(test), allow(dead_code))]
    TowardZero,
}

/// Deliberate faults in the rotation fold, used only by tests to show each
/// one changes the package digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FoldFault {
    None,
    #[cfg_attr(not(test), allow(dead_code))]
    SwapSide,
    #[cfg_attr(not(test), allow(dead_code))]
    SkipTensor(String),
}

/// A rotated tensor in f64, before rounding.
struct Folded {
    shape: Vec<u64>,
    values: Vec<f64>,
}

/// Regenerate a manifest rotation matrix and check it against its pinned
/// digest.
pub fn load_rotation(manifest: &Manifest, name: &str) -> Result<Vec<f64>> {
    let rotation = manifest
        .rotations
        .get(name)
        .ok_or_else(|| perr!("unknown rotation `{name}`"))?;
    hadamard::regenerate_checked(name, rotation)
}

/// Convert one profile from a loaded checkpoint.
pub fn convert_profile(
    manifest: &Manifest,
    profile_id: &str,
    checkpoint: &Checkpoint,
) -> Result<Vec<u8>> {
    let profile = manifest
        .profiles
        .get(profile_id)
        .ok_or_else(|| perr!("unknown profile `{profile_id}`"))?;
    let rotation = match profile.rotation.as_deref() {
        None | Some("none") => None,
        Some(name) => Some(load_rotation(manifest, name)?),
    };
    convert_with(
        manifest.model(&profile.model)?,
        profile_id,
        profile,
        checkpoint,
        rotation.as_deref(),
        Rounding::NearestEven,
        &FoldFault::None,
    )
}

/// Convert from a checkpoint file on disk and return the package bytes.
pub fn convert_profile_file(
    manifest: &Manifest,
    profile_id: &str,
    checkpoint_path: &Path,
) -> Result<Vec<u8>> {
    let profile = manifest
        .profiles
        .get(profile_id)
        .ok_or_else(|| perr!("unknown profile `{profile_id}`"))?;
    let model = manifest.model(&profile.model)?;
    let pinned = model
        .files
        .get("model.safetensors")
        .ok_or_else(|| perr!("{}: no pinned model.safetensors digest", profile.model))?;
    let actual = sha256_file(checkpoint_path)
        .map_err(|error| perr!("hash {}: {error}", checkpoint_path.display()))?;
    if &actual != pinned {
        return Err(perr!(
            "{} has SHA-256 {actual}, manifest pins {pinned}",
            checkpoint_path.display()
        ));
    }
    let bytes = std::fs::read(checkpoint_path)
        .map_err(|error| perr!("read {}: {error}", checkpoint_path.display()))?;
    convert_profile(manifest, profile_id, &Checkpoint::from_bytes(bytes)?)
}

pub(crate) fn convert_with(
    model: &Model,
    profile_id: &str,
    profile: &Profile,
    checkpoint: &Checkpoint,
    rotation: Option<&[f64]>,
    rounding: Rounding,
    fault: &FoldFault,
) -> Result<Vec<u8>> {
    if !profile.lane.is_worker() {
        return Err(perr!(
            "profile `{profile_id}` is in-process Metal and has no converted package"
        ));
    }
    let expected = expected_tensors(model)?;
    for (name, shape) in &expected {
        let info = checkpoint.info(name)?;
        if &info.shape != shape {
            return Err(perr!(
                "checkpoint tensor `{name}` has shape {:?}, expected {shape:?}",
                info.shape
            ));
        }
    }
    if let Some(extra) = checkpoint
        .header
        .tensors
        .keys()
        .find(|name| !expected.contains_key(*name))
    {
        return Err(perr!("checkpoint has unexpected tensor `{extra}`"));
    }

    let mut folded: BTreeMap<String, Folded> = BTreeMap::new();
    let mut dropped: Vec<String> = Vec::new();
    match (profile.rotation.as_deref(), rotation) {
        (None | Some("none"), None) => {}
        (Some(MODERNBERT_ROTATION), Some(q)) => {
            if model.architecture.family != Family::Modernbert {
                return Err(perr!(
                    "rotation `{MODERNBERT_ROTATION}` applies only to ModernBERT"
                ));
            }
            fold_modernbert(model, checkpoint, q, fault, &mut folded, &mut dropped)?;
        }
        (name, matrix) => {
            return Err(perr!(
                "profile `{profile_id}` rotation {name:?} with matrix present = {}",
                matrix.is_some()
            ))
        }
    }

    let mut names: Vec<String> = expected
        .keys()
        .filter(|name| !dropped.contains(name))
        .cloned()
        .collect();
    for name in folded.keys() {
        if !names.contains(name) {
            names.push(name.clone());
        }
    }
    if let Fp32Tensors::List(kept) = &profile.fp32_tensors {
        if let Some(missing) = kept.iter().find(|name| !names.contains(name)) {
            return Err(perr!("fp32 tensor `{missing}` is not in the package"));
        }
    }

    let mut tensors = BTreeMap::new();
    for name in names {
        let fp32 = match &profile.fp32_tensors {
            Fp32Tensors::All(_) => true,
            Fp32Tensors::List(kept) => kept.contains(&name),
        };
        let tensor = match folded.get(&name) {
            Some(rotated) => {
                let values: Vec<f32> = rotated.values.iter().map(|v| *v as f32).collect();
                encode(&name, rotated.shape.clone(), &values, fp32, rounding)?
            }
            None => {
                let (info, raw) = checkpoint.raw(&name)?;
                if fp32 && info.dtype == StDType::F32 {
                    PackageTensor {
                        dtype: StDType::F32,
                        shape: info.shape.clone(),
                        bytes: raw.to_vec(),
                    }
                } else if !fp32 && info.dtype == StDType::F16 && rounding == Rounding::NearestEven {
                    // f16 to f16 is the identity under any rounding mode that
                    // returns representable values unchanged.
                    PackageTensor {
                        dtype: StDType::F16,
                        shape: info.shape.clone(),
                        bytes: raw.to_vec(),
                    }
                } else {
                    encode(
                        &name,
                        info.shape.clone(),
                        &checkpoint.values_f32(&name)?,
                        fp32,
                        rounding,
                    )?
                }
            }
        };
        tensors.insert(name, tensor);
    }
    Ok(write_package(profile_id, &tensors))
}

fn encode(
    name: &str,
    shape: Vec<u64>,
    values: &[f32],
    fp32: bool,
    rounding: Rounding,
) -> Result<PackageTensor> {
    if fp32 {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        return Ok(PackageTensor {
            dtype: StDType::F32,
            shape,
            bytes,
        });
    }
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for &value in values {
        let half = round_f16(value, rounding);
        if value.is_finite() && !half.is_finite() {
            return Err(perr!("tensor `{name}` value {value} overflows f16"));
        }
        bytes.extend(half.to_le_bytes());
    }
    Ok(PackageTensor {
        dtype: StDType::F16,
        shape,
        bytes,
    })
}

fn round_f16(value: f32, rounding: Rounding) -> half::f16 {
    let nearest = half::f16::from_f32(value);
    match rounding {
        Rounding::NearestEven => nearest,
        Rounding::TowardZero => {
            if nearest.is_finite() && nearest.to_f32().abs() > value.abs() {
                // One step toward zero in magnitude.
                half::f16::from_bits(nearest.to_bits() - 1)
            } else {
                nearest
            }
        }
    }
}

/// `a` (`m × k`) times `b` (`k × n`), row-major. Each output element sums its
/// `k` products in ascending `k` order, starting from zero; the loop order
/// keeps that order whatever the compiler vectorizes, so results are
/// reproducible bit for bit.
pub(crate) fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        let out = &mut c[i * n..(i + 1) * n];
        for p in 0..k {
            let scale = a[i * k + p];
            let row = &b[p * n..(p + 1) * n];
            for j in 0..n {
                out[j] += scale * row[j];
            }
        }
    }
    c
}

fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; a.len()];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

fn values_f64(checkpoint: &Checkpoint, name: &str) -> Result<(Vec<u64>, Vec<f64>)> {
    let shape = checkpoint.info(name)?.shape.clone();
    let values = checkpoint
        .values_f32(name)?
        .into_iter()
        .map(f64::from)
        .collect();
    Ok((shape, values))
}

/// Subtract each row's mean (summed in ascending column order).
fn center_rows(values: &mut [f64], cols: usize) {
    for row in values.chunks_exact_mut(cols) {
        let mean = row.iter().sum::<f64>() / cols as f64;
        row.iter_mut().for_each(|v| *v -= mean);
    }
}

/// Subtract each column's mean over the rows (summed in ascending row order).
fn center_columns(values: &mut [f64], rows: usize, cols: usize) {
    for j in 0..cols {
        let mean = (0..rows).map(|i| values[i * cols + j]).sum::<f64>() / rows as f64;
        for i in 0..rows {
            values[i * cols + j] -= mean;
        }
    }
}

fn scale_columns(values: &mut [f64], cols: usize, gamma: &[f64]) {
    for row in values.chunks_exact_mut(cols) {
        row.iter_mut().zip(gamma).for_each(|(v, g)| *v *= g);
    }
}

fn fold_modernbert(
    model: &Model,
    checkpoint: &Checkpoint,
    q: &[f64],
    fault: &FoldFault,
    folded: &mut BTreeMap<String, Folded>,
    dropped: &mut Vec<String>,
) -> Result<()> {
    let p = model.tensor_prefix.as_str();
    let d = model.architecture.int("hidden_size")? as usize;
    let layers = model.architecture.int("num_hidden_layers")?;
    if q.len() != d * d {
        return Err(perr!("rotation matrix is not {d}x{d}"));
    }
    let qt = transpose(q, d, d);
    let gamma = |name: &str| -> Result<Vec<f64>> { Ok(values_f64(checkpoint, name)?.1) };
    let skip = |name: &str| matches!(fault, FoldFault::SkipTensor(skipped) if skipped == name);
    // `W Q`. The side-swap fault computes `(Q Wᵀ)ᵀ = W Qᵀ` instead: the same
    // product with Q applied from the other side of the transposed weight.
    let right_q = |values: &[f64], rows: usize| -> Vec<f64> {
        if *fault == FoldFault::SwapSide {
            matmul(values, rows, d, &qt, d)
        } else {
            matmul(values, rows, d, q, d)
        }
    };
    let mut put = |name: String, shape: Vec<u64>, values: Vec<f64>| {
        if !skip(&name) {
            folded.insert(name, Folded { shape, values });
        }
    };

    let embedding_norm = format!("{p}embeddings.norm.weight");
    let gamma_embedding = gamma(&embedding_norm)?;

    let embedding = format!("{p}embeddings.tok_embeddings.weight");
    let (shape, mut values) = values_f64(checkpoint, &embedding)?;
    center_rows(&mut values, d);
    let rows = shape[0] as usize;
    put(embedding, shape, right_q(&values, rows));

    for layer in 0..layers {
        let l = format!("{p}layers.{layer}");
        let attention_gamma = if layer == 0 {
            gamma_embedding.clone()
        } else {
            gamma(&format!("{l}.attn_norm.weight"))?
        };
        for (weight, norm_gamma) in [
            (format!("{l}.attn.Wqkv.weight"), attention_gamma),
            (
                format!("{l}.mlp.Wi.weight"),
                gamma(&format!("{l}.mlp_norm.weight"))?,
            ),
        ] {
            let (shape, mut values) = values_f64(checkpoint, &weight)?;
            scale_columns(&mut values, d, &norm_gamma);
            let rows = shape[0] as usize;
            put(weight, shape, right_q(&values, rows));
        }
        for weight in [format!("{l}.attn.Wo.weight"), format!("{l}.mlp.Wo.weight")] {
            let (shape, mut values) = values_f64(checkpoint, &weight)?;
            let cols = shape[1] as usize;
            center_columns(&mut values, d, cols);
            // `Qᵀ W`; the side-swap fault computes `(Wᵀ Qᵀ)ᵀ = Q W` instead.
            let rotated = if *fault == FoldFault::SwapSide {
                transpose(
                    &matmul(&transpose(&values, d, cols), cols, d, &qt, d),
                    cols,
                    d,
                )
            } else {
                matmul(&qt, d, d, &values, cols)
            };
            put(weight, shape, rotated);
        }
        if layer > 0 {
            dropped.push(format!("{l}.attn_norm.weight"));
        }
        dropped.push(format!("{l}.mlp_norm.weight"));
    }

    // rotation_in = Qᵀ · C · diag(γ_embedding) · Q
    let mut c_gamma = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let centering = if i == j { 1.0 } else { 0.0 } - 1.0 / d as f64;
            c_gamma[i * d + j] = centering * gamma_embedding[j];
        }
    }
    let rotation_in = matmul(&matmul(&qt, d, d, &c_gamma, d), d, d, q, d);
    put(
        ROTATION_IN_KEY.to_string(),
        vec![d as u64, d as u64],
        rotation_in,
    );

    // rotation_out = diag(γ_final) · Q
    let final_norm = format!("{p}final_norm.weight");
    let gamma_final = gamma(&final_norm)?;
    let mut rotation_out = q.to_vec();
    for (i, row) in rotation_out.chunks_exact_mut(d).enumerate() {
        row.iter_mut().for_each(|v| *v *= gamma_final[i]);
    }
    put(
        ROTATION_OUT_KEY.to_string(),
        vec![d as u64, d as u64],
        rotation_out,
    );

    dropped.push(embedding_norm);
    dropped.push(final_norm);
    // A skipped in-scope tensor is neither rotated nor dropped: it passes
    // through from the checkpoint unrotated.
    if let FoldFault::SkipTensor(name) = fault {
        dropped.retain(|d| d != name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::sha256_hex;
    use crate::manifest::{Manifest, MANIFEST_FILE};
    use crate::safetensors::{verify_package_structure, write_package_with, Header};
    use serde_json::json;

    fn manifest() -> Manifest {
        Manifest::load(&crate::parity_dir().join(MANIFEST_FILE)).unwrap()
    }

    /// gte-modernbert-base shrunk to two layers of width 12, so the full
    /// conversion, rotation included, runs in milliseconds.
    fn tiny_model(manifest: &Manifest) -> Model {
        let mut model = manifest.model("gte-modernbert-base").unwrap().clone();
        let params = &mut model.architecture.params;
        params.insert("num_hidden_layers".into(), json!(2));
        params.insert("hidden_size".into(), json!(12));
        params.insert("intermediate_size".into(), json!(6));
        params.insert("vocab_size".into(), json!(5));
        model
    }

    fn encode_values(values: &[f32], dtype: StDType) -> Vec<u8> {
        match dtype {
            StDType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            StDType::F16 => values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect(),
            StDType::Bf16 => values
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect(),
        }
    }

    /// A checkpoint holding every tensor the tiny model needs, with
    /// deterministic values. The first two elements of every non-norm tensor
    /// are chosen to separate rounding modes when the source is f32.
    fn tiny_checkpoint(model: &Model, dtype: StDType) -> Checkpoint {
        let mut tensors = BTreeMap::new();
        for (index, (name, shape)) in expected_tensors(model).unwrap().into_iter().enumerate() {
            let count = shape.iter().product::<u64>() as usize;
            let values: Vec<f32> = (0..count)
                .map(|i| {
                    if name.ends_with("norm.weight") {
                        1.0 + (i % 7) as f32 / 16.0
                    } else if i == 0 {
                        // Halfway between two f16 values: nearest-even rounds
                        // down, as round-toward-zero does.
                        1.0 + 2f32.powi(-11)
                    } else if i == 1 {
                        // Above halfway: nearest-even rounds up, toward-zero
                        // rounds down.
                        1.0 + 2f32.powi(-11) + 2f32.powi(-13)
                    } else {
                        ((i * 37 + index * 11) % 101) as f32 / 50.0 - 1.0
                    }
                })
                .collect();
            tensors.insert(
                name,
                PackageTensor {
                    dtype,
                    shape,
                    bytes: encode_values(&values, dtype),
                },
            );
        }
        let bytes = write_package_with(&[("format", "pt")], tensors.iter().collect());
        Checkpoint::from_bytes(bytes).unwrap()
    }

    /// A 12x12 orthogonal matrix: the order-12 Hadamard matrix with random
    /// column signs, scaled by 1/sqrt(12). It is not symmetric, so applying
    /// it from the wrong side gives a different result.
    fn tiny_rotation() -> Vec<f64> {
        let h12: [[i8; 12]; 12] = [
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
            [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1],
            [1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
            [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1],
            [1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
            [1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1],
            [1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
            [1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1],
            [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
            [1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1],
            [1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1],
        ];
        let signs = [1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1];
        let scale = 1.0 / 12f64.sqrt();
        let mut q = Vec::with_capacity(144);
        for row in h12 {
            for (j, value) in row.into_iter().enumerate() {
                q.push(f64::from(value) * f64::from(signs[j]) * scale);
            }
        }
        q
    }

    fn header_of(bytes: &[u8]) -> Header {
        let (header, _) = crate::safetensors::split_file(bytes).unwrap();
        crate::safetensors::parse_header_json(header).unwrap()
    }

    struct Fixture {
        model: Model,
        profile: Profile,
        id: String,
    }

    fn fixture(profile_id: &str) -> Fixture {
        let manifest = manifest();
        let model = tiny_model(&manifest);
        let profile = manifest.profiles[profile_id].clone();
        Fixture {
            model,
            profile,
            id: profile_id.to_string(),
        }
    }

    fn convert_fixture(
        f: &Fixture,
        checkpoint: &Checkpoint,
        q: Option<&[f64]>,
        rounding: Rounding,
        fault: &FoldFault,
    ) -> Vec<u8> {
        convert_with(&f.model, &f.id, &f.profile, checkpoint, q, rounding, fault).unwrap()
    }

    #[test]
    fn the_rotation_fixture_is_orthogonal() {
        let q = tiny_rotation();
        let product = matmul(&transpose(&q, 12, 12), 12, 12, &q, 12);
        for i in 0..12 {
            for j in 0..12 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product[i * 12 + j] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn packages_are_canonical_and_reproducible() {
        let f = fixture("gte-modernbert-base.owned-cuda");
        let checkpoint = tiny_checkpoint(&f.model, StDType::F32);
        let a = convert_fixture(
            &f,
            &checkpoint,
            None,
            Rounding::NearestEven,
            &FoldFault::None,
        );
        let b = convert_fixture(
            &f,
            &checkpoint,
            None,
            Rounding::NearestEven,
            &FoldFault::None,
        );
        assert_eq!(a, b);
        verify_package_structure(&a, &f.id).unwrap();
        let header = header_of(&a);
        assert_eq!(header.key_order[0], "__metadata__");
        let names: Vec<&String> = header.key_order[1..].iter().collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        assert!(header.tensors.values().all(|t| t.dtype == StDType::F16));
    }

    #[test]
    fn reordered_tensors_fail() {
        let f = fixture("gte-modernbert-base.owned-cuda");
        let checkpoint = tiny_checkpoint(&f.model, StDType::F32);
        let canonical = convert_fixture(
            &f,
            &checkpoint,
            None,
            Rounding::NearestEven,
            &FoldFault::None,
        );
        let (header, data) = crate::safetensors::split_file(&canonical).unwrap();
        let parsed = crate::safetensors::parse_header_json(header).unwrap();
        let mut tensors: Vec<(String, PackageTensor)> = parsed
            .tensors
            .iter()
            .map(|(name, info)| {
                let bytes = data[info.data_offsets.0..info.data_offsets.1].to_vec();
                (
                    name.clone(),
                    PackageTensor {
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                        bytes,
                    },
                )
            })
            .collect();
        tensors.reverse();
        let reordered = write_package_with(
            &[("conversion_rule", "v1"), ("profile", f.id.as_str())],
            tensors
                .iter()
                .map(|(name, tensor)| (name, tensor))
                .collect(),
        );
        assert_ne!(sha256_hex(&reordered), sha256_hex(&canonical));
        let error = verify_package_structure(&reordered, &f.id).unwrap_err();
        assert!(error.0.contains("lexicographic"), "{error}");
    }

    #[test]
    fn an_extra_metadata_key_fails() {
        let f = fixture("gte-modernbert-base.owned-cuda");
        let checkpoint = tiny_checkpoint(&f.model, StDType::F32);
        let canonical = convert_fixture(
            &f,
            &checkpoint,
            None,
            Rounding::NearestEven,
            &FoldFault::None,
        );
        let (header, data) = crate::safetensors::split_file(&canonical).unwrap();
        let parsed = crate::safetensors::parse_header_json(header).unwrap();
        let tensors: BTreeMap<String, PackageTensor> = parsed
            .tensors
            .iter()
            .map(|(name, info)| {
                let bytes = data[info.data_offsets.0..info.data_offsets.1].to_vec();
                (
                    name.clone(),
                    PackageTensor {
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                        bytes,
                    },
                )
            })
            .collect();
        let extra = write_package_with(
            &[
                ("conversion_rule", "v1"),
                ("format", "pt"),
                ("profile", f.id.as_str()),
            ],
            tensors.iter().collect(),
        );
        assert_ne!(sha256_hex(&extra), sha256_hex(&canonical));
        let error = verify_package_structure(&extra, &f.id).unwrap_err();
        assert!(error.0.contains("__metadata__"), "{error}");
        // The canonical bytes are exactly what write_package produces.
        assert_eq!(write_package(&f.id, &tensors), canonical);
    }

    #[test]
    fn f16_rounding_is_nearest_even_and_toward_zero_changes_the_digest() {
        let f = fixture("gte-modernbert-base.owned-cuda");
        let checkpoint = tiny_checkpoint(&f.model, StDType::F32);
        let nearest = convert_fixture(
            &f,
            &checkpoint,
            None,
            Rounding::NearestEven,
            &FoldFault::None,
        );
        let toward_zero = convert_fixture(
            &f,
            &checkpoint,
            None,
            Rounding::TowardZero,
            &FoldFault::None,
        );
        assert_ne!(sha256_hex(&nearest), sha256_hex(&toward_zero));
        // Both are structurally valid: only the reconverted digest catches a
        // wrong rounding mode.
        verify_package_structure(&toward_zero, &f.id).unwrap();

        let package = Checkpoint::from_bytes(nearest).unwrap();
        let values = package.values_f32("layers.0.attn.Wo.weight").unwrap();
        // Exactly halfway: ties to the even neighbour, 1.0.
        assert_eq!(values[0], 1.0);
        // Above halfway: rounds up to the next f16, 1 + 2^-10.
        assert_eq!(values[1], 1.0 + 2f32.powi(-10));
        let rtz = Checkpoint::from_bytes(toward_zero).unwrap();
        assert_eq!(rtz.values_f32("layers.0.attn.Wo.weight").unwrap()[1], 1.0);
    }

    #[test]
    fn fp32_kept_tensors_are_copied_bit_exactly() {
        let mut f = fixture("gte-modernbert-base.owned-cuda");
        let kept = "layers.1.mlp.Wo.weight".to_string();
        f.profile.fp32_tensors = Fp32Tensors::List(vec![kept.clone()]);
        let checkpoint = tiny_checkpoint(&f.model, StDType::F32);
        let package = Checkpoint::from_bytes(convert_fixture(
            &f,
            &checkpoint,
            None,
            Rounding::NearestEven,
            &FoldFault::None,
        ))
        .unwrap();
        let (info, bytes) = package.raw(&kept).unwrap();
        assert_eq!(info.dtype, StDType::F32);
        assert_eq!(bytes, checkpoint.raw(&kept).unwrap().1);
    }

    #[test]
    fn f16_overflow_is_an_error() {
        let error = encode("t", vec![1], &[70000.0], false, Rounding::NearestEven).unwrap_err();
        assert!(error.0.contains("overflows f16"), "{error}");
    }

    #[test]
    fn rotated_package_drops_folded_norms_and_adds_both_projections() {
        let f = fixture("gte-modernbert-base.ane-direct-worker");
        let checkpoint = tiny_checkpoint(&f.model, StDType::F16);
        let q = tiny_rotation();
        let bytes = convert_fixture(
            &f,
            &checkpoint,
            Some(&q),
            Rounding::NearestEven,
            &FoldFault::None,
        );
        verify_package_structure(&bytes, &f.id).unwrap();
        let header = header_of(&bytes);
        for added in [
            ROTATION_IN_KEY,
            ROTATION_OUT_KEY,
            "embeddings.tok_embeddings.weight",
        ] {
            assert_eq!(header.tensors[added].dtype, StDType::F32, "{added}");
        }
        assert!(header
            .tensors
            .keys()
            .all(|name| !name.ends_with("norm.weight")));
        assert_eq!(
            header.tensors["layers.0.attn.Wqkv.weight"].dtype,
            StDType::F16
        );
    }

    #[test]
    fn each_rotation_mutation_changes_the_package_digest() {
        let f = fixture("gte-modernbert-base.ane-direct-worker");
        let checkpoint = tiny_checkpoint(&f.model, StDType::F16);
        let q = tiny_rotation();
        let digest = |q: &[f64], fault: &FoldFault| {
            sha256_hex(&convert_fixture(
                &f,
                &checkpoint,
                Some(q),
                Rounding::NearestEven,
                fault,
            ))
        };
        let baseline = digest(&q, &FoldFault::None);

        let mut changed = q.clone();
        let bits = changed[5].to_bits() ^ (1 << 40);
        changed[5] = f64::from_bits(bits);
        assert_ne!(digest(&changed, &FoldFault::None), baseline, "matrix byte");

        assert_ne!(
            digest(&q, &FoldFault::SwapSide),
            baseline,
            "multiplication side"
        );

        for skipped in [
            "embeddings.tok_embeddings.weight",
            "layers.0.attn.Wqkv.weight",
            "layers.1.attn.Wo.weight",
            "layers.1.mlp.Wi.weight",
            "layers.0.mlp.Wo.weight",
            "layers.1.attn_norm.weight",
            "final_norm.weight",
        ] {
            assert_ne!(
                digest(&q, &FoldFault::SkipTensor(skipped.to_string())),
                baseline,
                "skipping {skipped}"
            );
        }
        // Skipping one of the two new projections leaves an fp32 tensor the
        // profile requires out of the package, which fails conversion.
        for skipped in [ROTATION_IN_KEY, ROTATION_OUT_KEY] {
            let fault = FoldFault::SkipTensor(skipped.to_string());
            assert!(convert_with(
                &f.model,
                &f.id,
                &f.profile,
                &checkpoint,
                Some(&q),
                Rounding::NearestEven,
                &fault
            )
            .is_err());
        }
    }

    #[test]
    fn qwen3_and_unrotated_profiles_refuse_a_matrix() {
        let f = fixture("gte-modernbert-base.owned-cuda");
        let checkpoint = tiny_checkpoint(&f.model, StDType::F16);
        let q = tiny_rotation();
        assert!(convert_with(
            &f.model,
            &f.id,
            &f.profile,
            &checkpoint,
            Some(&q),
            Rounding::NearestEven,
            &FoldFault::None
        )
        .is_err());
    }
}
