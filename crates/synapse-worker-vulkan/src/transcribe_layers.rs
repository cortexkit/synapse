//! Test-only comparison with hash-addressed fp32 and fp16-storage CPU captures.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::{Path, PathBuf},
};
use synapse_parity::speech::Artifact;

pub(crate) fn compare(reference: &[f32], actual: &[f32], per_frame: bool) -> Result<Value> {
    ensure!(
        !reference.is_empty()
            && reference.len() == actual.len()
            && reference.len().is_multiple_of(1024),
        "layer shape mismatch"
    );
    ensure!(
        reference.iter().chain(actual).all(|x| x.is_finite()),
        "nonfinite layer capture"
    );
    let mut max_abs = 0f64;
    let mut minimum = 1f64;
    let mut worst = 0;
    let mut bad = Vec::new();
    let mut reference_norms = Vec::new();
    let mut actual_norms = Vec::new();
    let mut cosines = Vec::new();
    for (index, (r, a)) in reference
        .as_chunks::<1024>()
        .0
        .iter()
        .zip(actual.as_chunks::<1024>().0)
        .enumerate()
    {
        let (mut dot, mut rr, mut aa) = (0f64, 0f64, 0f64);
        for (&r, &a) in r.iter().zip(a) {
            let (r, a) = (f64::from(r), f64::from(a));
            max_abs = max_abs.max((r - a).abs());
            dot += r * a;
            rr += r * r;
            aa += a * a;
        }
        let (rn, an) = (rr.sqrt(), aa.sqrt());
        let cosine = dot / (rn * an);
        cosines.push(cosine.is_finite().then_some(cosine));
        if !cosine.is_finite() || cosine < minimum {
            minimum = cosine;
            worst = index;
        }
        if !cosine.is_finite() || cosine < 0.999 {
            bad.push(json!({"frame":index,"frames_from_end":reference.len()/1024-1-index,
                "cosine":cosine.is_finite().then_some(cosine),"reference_norm":rn,"actual_norm":an}));
        }
        reference_norms.push(rn);
        actual_norms.push(an);
    }
    let mut ordered: Vec<_> = cosines.iter().flatten().copied().collect();
    ordered.sort_by(f64::total_cmp);
    let quantile =
        |p: f64| (!ordered.is_empty()).then(|| ordered[(p * (ordered.len() - 1) as f64) as usize]);
    let mut row = json!({"max_abs":max_abs,"minimum_cosine":minimum.is_finite().then_some(minimum),
        "cosine_quantiles":{"min":quantile(0.),"p01":quantile(0.01),"p50":quantile(0.5),"p99":quantile(0.99),"max":quantile(1.)},
        "cosine_below_0_99999_frames":cosines.iter().filter(|c|c.is_none_or(|c|c<0.99999)).count(),
        "worst_frame":worst,"bad_frames":bad,"reference_norm_at_worst":reference_norms[worst],"actual_norm_at_worst":actual_norms[worst],
        "reference_norm_min":reference_norms.iter().copied().fold(f64::INFINITY,f64::min),
        "reference_norm_max":reference_norms.iter().copied().fold(0.,f64::max),
        "actual_norm_min":actual_norms.iter().copied().fold(f64::INFINITY,f64::min),
        "actual_norm_max":actual_norms.iter().copied().fold(0.,f64::max)});
    if per_frame {
        row["reference_frame_norms"] = json!(reference_norms);
        row["actual_frame_norms"] = json!(actual_norms);
        row["frame_cosines"] = json!(cosines);
    }
    Ok(row)
}

struct Reference {
    mode: &'static str,
    directory: PathBuf,
    layers: BTreeMap<String, Artifact>,
}

fn artifact(row: &Value) -> Result<Artifact> {
    Ok(Artifact {
        path: Some(
            row["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing capture path"))?
                .into(),
        ),
        cache_path: None,
        sha256: row["sha256"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing capture SHA"))?
            .into(),
        bytes: row["bytes"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("missing capture byte count"))?,
    })
}

fn references(root: &Path, id: &str) -> Result<Vec<Reference>> {
    let mut out = Vec::new();
    for mode in ["fp32", "f16-storage"] {
        let directory = root.join(mode).join(id);
        let index = directory.join("layers.json");
        if !index.exists() {
            continue;
        }
        let index: BTreeMap<String, Value> = serde_json::from_slice(&std::fs::read(index)?)?;
        let layers = index
            .into_iter()
            .map(|(name, row)| -> Result<_> {
                ensure!(
                    name == "subsampling" || (0..24).any(|i| name == format!("layer-{i:02}")),
                    "unknown layer capture name"
                );
                Ok((name, artifact(&row)?))
            })
            .collect::<Result<_>>()?;
        out.push(Reference {
            mode,
            directory,
            layers,
        });
    }
    Ok(out)
}

pub(crate) fn trace_agreement(
    id: &str,
    actual: &synapse_parity::speech::SpeechObservation,
) -> Result<Option<Value>> {
    let Some(root) = std::env::var_os("SYNAPSE_VULKAN_SPEECH_LAYER_REFERENCE") else {
        return Ok(None);
    };
    let directory = Path::new(&root).join("f16-storage").join(id);
    let index = directory.join("traces.json");
    if !index.exists() {
        return Ok(None);
    }
    let index: Value = serde_json::from_slice(&std::fs::read(index)?)?;
    let bytes = artifact(&index["f16-storage-joint"])?.read(&directory, &directory)?;
    let shadow = synapse_parity::speech::decode_trace(&bytes)?;
    let frames = actual
        .valid_encoder_frames
        .ok_or_else(|| anyhow::anyhow!("missing encoder frames"))?;
    let grade =
        synapse_parity::speech::grade_trace(&shadow, &actual.trace, &actual.triples, frames, 0.);
    let triples_equal = synapse_parity::speech::implied_triples(&shadow, frames)? == actual.triples;
    let first_difference = (0..shadow.len().max(actual.trace.len())).find(|&i| {
        match (shadow.get(i), actual.trace.get(i)) {
            (Some(r), Some(a)) => {
                r.frame_index != a.frame_index
                    || r.token_top1 != a.token_top1
                    || r.duration_top1 != a.duration_top1
            }
            _ => true,
        }
    });
    let logit_max_abs = (shadow.len() == actual.trace.len()).then(|| {
        shadow
            .iter()
            .zip(&actual.trace)
            .flat_map(|(r, a)| {
                [
                    (r.token_logit1 - a.token_logit1).abs(),
                    (r.token_logit2 - a.token_logit2).abs(),
                    (r.duration_logit1 - a.duration_logit1).abs(),
                    (r.duration_logit2 - a.duration_logit2).abs(),
                ]
            })
            .fold(0f32, f32::max)
    });
    Ok(Some(
        json!({"fixture":id,"exact_decisions":grade.is_ok(),"triples_identical":triples_equal,
        "first_differing_step":first_difference,"shadow_steps":shadow.len(),"vulkan_steps":actual.trace.len(),
        "logit_max_abs":logit_max_abs,"error":grade.err().map(|e|e.to_string())}),
    ))
}

pub(crate) fn attach(
    engine: &mut crate::runtime::transcribe::Engine,
    id: &str,
    evidence: &Path,
) -> Result<()> {
    engine.debug_clear_capture();
    let Some(root) = std::env::var_os("SYNAPSE_VULKAN_SPEECH_LAYER_REFERENCE") else {
        return Ok(());
    };
    let references = references(Path::new(&root), id)?;
    let names: BTreeSet<_> = references
        .iter()
        .flat_map(|r| r.layers.keys().cloned())
        .collect();
    if names.is_empty() {
        return Ok(());
    }
    let id = id.to_owned();
    let path = evidence.join(format!("{id}-layer-errors.jsonl"));
    std::fs::File::create(&path)?;
    engine.debug_capture_layers(names, move |name, values| {
        for reference in &references {
            if let Some(artifact) = reference.layers.get(name) {
                let bytes = artifact.read(&reference.directory, &reference.directory)?;
                let expected = synapse_core::decode_f32_frame(&bytes)?;
                let row = compare(&expected, values, id == "silence-30s" || name == "layer-23")?;
                let finding =
                    json!({"fixture":id,"layer":name,"reference":reference.mode,"metrics":row});
                let mut output = std::fs::OpenOptions::new().append(true).open(&path)?;
                writeln!(output, "{finding}")?;
                println!(
                    "{id} {name} vs {} max_abs={} minimum_cosine={}",
                    reference.mode, row["max_abs"], row["minimum_cosine"]
                );
            }
        }
        Ok(())
    });
    Ok(())
}

#[test]
fn layer_comparison_reports_interior_frames_and_nonzero_norms() {
    let reference = vec![1f32; 3 * 1024];
    let mut actual = reference.clone();
    actual[1024..2048].fill(-1.);
    let row = compare(&reference, &actual, true).unwrap();
    assert_eq!(row["max_abs"], 2.);
    assert_eq!(row["minimum_cosine"], -1.);
    assert_eq!(row["bad_frames"].as_array().unwrap().len(), 1);
    assert_eq!(row["bad_frames"][0]["frame"], 1);
    assert_eq!(row["bad_frames"][0]["frames_from_end"], 1);
    assert_eq!(row["reference_norm_at_worst"], 32.);
    assert_eq!(row["actual_frame_norms"], json!([32., 32., 32.]));
    assert!(compare(&reference, &actual[..1024], false).is_err());
}

#[test]
fn layer_comparison_marks_zero_norms_without_serializing_nonfinite_numbers() {
    let row = compare(&vec![0.; 1024], &vec![0.; 1024], false).unwrap();
    assert!(row["minimum_cosine"].is_null());
    assert_eq!(row["bad_frames"].as_array().unwrap().len(), 1);
    assert_eq!(row["reference_norm_min"], 0.);
    assert!(!serde_json::to_string(&row).unwrap().contains("NaN"));
}
