//! Owner-run probes use identical CPU component-boundary input tensors and controlled padding.
use super::*;
use crate::parakeet::{attention, convolution, feed_forward, norm, HIDDEN};
use ane::{Graph, PadMode, TensorData};

fn component_graph(model: &Model, shape: usize, layer: usize, component: &str) -> Result<Graph> {
    ensure!(
        [22, 23].contains(&layer),
        "only late-layer diagnostics are supported"
    );
    let mut g = Graph::new();
    let x = g.placeholder(crate::modernbert::shape(shape, HIDDEN));
    let base = format!("encoder.layers.{layer}");
    if component.starts_with("norm_") {
        ensure!(
            [
                "norm_feed_forward1",
                "norm_self_att",
                "norm_conv",
                "norm_feed_forward2",
                "norm_out"
            ]
            .contains(&component),
            "unknown normalization seam"
        );
        norm(&mut g, x, model, &format!("{base}.{component}"))?;
    } else if component == "feed_forward1" || component == "feed_forward2" {
        feed_forward(
            &mut g,
            x,
            model,
            &base,
            if component.ends_with('2') { 2 } else { 1 },
        )?;
    } else {
        let valid = g.placeholder(crate::modernbert::shape(shape, 1));
        match component {
            "attention" => {
                attention(&mut g, x, valid, model, &base, shape)?;
            }
            "convolution" => {
                convolution(&mut g, x, valid, model, &base)?;
            }
            "block" => {
                let x = feed_forward(&mut g, x, model, &base, 1)?;
                let x = attention(&mut g, x, valid, model, &base, shape)?;
                let x = convolution(&mut g, x, valid, model, &base)?;
                let x = feed_forward(&mut g, x, model, &base, 2)?;
                norm(&mut g, x, model, &format!("{base}.norm_out"))?;
            }
            _ => anyhow::bail!("unknown component probe"),
        }
    }
    Ok(g)
}

fn channel_major(rows: &[f32], shape: usize) -> Result<Vec<f32>> {
    ensure!(
        rows.len() == shape * HIDDEN,
        "malformed full-rung seam input"
    );
    Ok((0..HIDDEN)
        .flat_map(|c| (0..shape).map(move |t| rows[t * HIDDEN + c]))
        .collect())
}

fn encoder_mask_probe(shape: usize, valid_mel: usize) -> Result<Value> {
    ane::autoreleasepool(|_| {
        let mut graph = Graph::new();
        let input_shape = crate::modernbert::shape(shape * 8, 1);
        let mut mask = graph.placeholder(input_shape);
        for _ in 0..3 {
            mask = graph.max_pool(mask, 1, 1, 1, 2, PadMode::Valid);
        }
        let executable = graph.compile(ane::NSQualityOfService::UserInteractive)?;
        let input = TensorData::new(input_shape);
        let output = TensorData::new(crate::modernbert::shape(shape, 1));
        let mut raw = vec![0.0; shape * 8];
        raw[..valid_mel].fill(1.0);
        input.copy_from_f32(&raw);
        executable.run(&[&input], &[&output])?;
        let actual = output.read_f32().to_vec();
        let expected: Vec<_> = (0..shape)
            .map(|t| if t < valid_mel.div_ceil(8) { 1.0 } else { 0.0 })
            .collect();
        Ok(json!({"expected": expected, "actual": actual, "exact": actual == expected}))
    })
}

pub(super) fn run(
    summary: &mut Value,
    manifest: &Manifest,
    snapshot: &Path,
    package: &Path,
    parity: &Path,
) -> Result<()> {
    let index_path = env_path("ANE_PARAKEET_COMPONENT_REFERENCE_INDEX")?;
    let directory = index_path
        .parent()
        .context("component reference directory")?;
    let index: Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
    let profile = Profile::select(PROFILE, "transcribe")?;
    ensure!(
        index["schema"] == "parakeet-component-reference-v1"
            && index["checkpoint_sha256"] == profile.model["files"]["model.safetensors"]
            && index["converted_package_digest"] == profile.numeric["converted_package_digest"],
        "component reference binding mismatch"
    );
    let fixture_id = index["fixture"].as_str().context("component fixture")?;
    let shape = index["rung"].as_u64().context("component rung")? as usize;
    let frames = index["valid_encoder_frames"]
        .as_u64()
        .context("valid component frames")? as usize;
    ensure!(
        shape == crate::backend::rung(frames)?,
        "component rung mismatch"
    );
    let fixtures = speech::load_transcribe_fixtures(parity, manifest)?;
    ensure!(
        fixtures.iter().any(|fixture| fixture.id == fixture_id),
        "component fixture is not sealed"
    );
    let document: Value = serde_json::from_slice(&std::fs::read(
        parity.join("fixtures/parakeet-tdt-0.6b-v3/reference.json"),
    )?)?;
    let case = document["cases"]
        .as_array()
        .context("fixture cases")?
        .iter()
        .find(|case| case["id"] == fixture_id)
        .context("component fixture missing")?;
    let mel = f32le(&artifact(
        &case["log_mel"],
        parity,
        &env_path("ANE_PARAKEET_MIRROR")?,
    )?)?;
    let mel_frames = case["mel_frames"].as_u64().context("mel frames")? as usize;
    let valid_mel = case["valid_mel_frames"]
        .as_u64()
        .context("valid mel frames")? as usize;
    let dump = env_path("ANE_PARAKEET_LAYER_DUMP_DIR")?;
    let mut model = load_model(manifest, profile, snapshot, package)?;
    super::super::validate_load(&model.profile, synapse_core::DEFAULT_MAX_FRAME_BYTES)?;
    let os = String::from_utf8(
        std::process::Command::new("sw_vers")
            .arg("-buildVersion")
            .output()?
            .stdout,
    )?;
    let mut progress = Progress::start();
    let admission = model.admit_progress(shape, os.trim(), &mut || progress.completed())?;
    progress.finish(600.0)?;
    crate::backend::validate_inventory(&admission.inventory, 24)?;
    ensure!(
        admission.inventory.executables.len() == 24,
        "warm shape executable count"
    );
    summary["mode"] = json!("component-accuracy-not-certification");
    summary["fixture"] = json!(fixture_id);
    summary["inventory"] = json!(admission.inventory);
    summary["diagnostic_peak_live_executables"] = json!(25);
    let mut progress = Progress::start();
    let warm = model.transcribe_progress(&mel, mel_frames, valid_mel, true, &mut || {
        progress.completed()
    })?;
    progress.finish(120.0)?;
    ensure!(warm.frames == frames, "warm valid-frame mismatch");
    summary["warm_encoder_gap_seconds"] = json!(progress.maximum);
    summary["programs"] = json!([]);
    save(summary)?;
    summary["encoder_mask_probe"] = encoder_mask_probe(shape, valid_mel)?;
    println!("PARAKEET_ENCODER_MASK {}", summary["encoder_mask_probe"]);
    let mut mismatch = summary["encoder_mask_probe"]["exact"] != true;
    for program in index["programs"].as_array().context("component programs")? {
        let layer = program["layer"].as_u64().context("component layer")? as usize;
        ensure!(
            [22, 23].contains(&layer),
            "only late-layer diagnostics are supported"
        );
        let component = program["component"].as_str().context("component name")?;
        let masked = ["attention", "convolution", "block"].contains(&component);
        ensure!(
            program["masked"] == masked,
            "component mask metadata mismatch"
        );
        let source = f32le(&artifact(&program["input"], directory, directory)?)?;
        let expected = f32le(&artifact(&program["expected"], directory, directory)?)?;
        let previous = f32le(&std::fs::read(
            dump.join(format!("layer-{}-rung-{shape}.f32le", layer - 1)),
        )?)?;
        ensure!(
            source.len() == shape * HIDDEN
                && previous.len() == source.len()
                && expected.len() == frames * HIDDEN,
            "component tensor shape mismatch"
        );
        let measurements = ane::autoreleasepool(|_| -> Result<Vec<Value>> {
            let graph = component_graph(&model, shape, layer, component)?;
            let (ops, input_shapes, output_shapes) = graph.diagnostic_shapes();
            let shapes = |shapes: Vec<ane::Shape>| {
                shapes
                    .into_iter()
                    .map(|s| [s.batch, s.channels, s.height, s.width])
                    .collect::<Vec<_>>()
            };
            let input_shapes = shapes(input_shapes);
            let output_shapes = shapes(output_shapes);
            let executable = graph.compile(ane::NSQualityOfService::UserInteractive)?;
            let input = TensorData::new(crate::modernbert::shape(shape, HIDDEN));
            let output = TensorData::new(crate::modernbert::shape(shape, HIDDEN));
            let valid = TensorData::new(crate::modernbert::shape(shape, 1));
            let mut mask = vec![0.0; shape];
            mask[..frames].fill(1.0);
            valid.copy_from_f32(&mask);
            let variants: &[&str] = if masked {
                &["cpu-padding", "ane-padding", "zero-padding"]
            } else {
                &["cpu-padding"]
            };
            let mut baseline = None;
            let mut results = Vec::new();
            for variant in variants {
                let mut rows = source.clone();
                if *variant == "ane-padding" {
                    rows[frames * HIDDEN..].copy_from_slice(&previous[frames * HIDDEN..]);
                } else if *variant == "zero-padding" {
                    rows[frames * HIDDEN..].fill(0.0);
                }
                input.copy_from_f32(&channel_major(&rows, shape)?);
                if masked {
                    executable.run_cached(&[&input, &valid], &[&output])?;
                } else {
                    executable.run_cached(&[&input], &[&output])?;
                }
                let raw = output.read_f32();
                let actual: Vec<_> = (0..frames)
                    .flat_map(|t| (0..HIDDEN).map(move |c| c * shape + t))
                    .map(|index| raw[index])
                    .collect();
                let distribution = encoder_cosine_distribution(&expected, &actual)?;
                let padding_max = rows[frames * HIDDEN..]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0, f32::max);
                let comparison = baseline
                    .as_ref()
                    .map(|baseline: &Vec<f32>| tensor_comparison(baseline, &actual));
                baseline.get_or_insert_with(|| actual.clone());
                let name = format!("component-layer-{layer}-{component}-{variant}.f32le");
                let raw = synapse_core::encode_f32_frame(&actual);
                std::fs::write(dump.join(&name), &raw)?;
                println!("PARAKEET_COMPONENT layer={layer} component={component} variant={variant} {distribution}");
                results.push(json!({"layer": layer, "component": component, "variant": variant,
                    "ops": ops, "inputs": input_shapes, "outputs": output_shapes,
                    "padding_max_abs": padding_max, "cosine": distribution,
                    "versus_cpu_padding": comparison, "max_abs_error": tensor_comparison(&expected, &actual),
                    "artifact": {"path": name, "bytes": raw.len(), "sha256": format!("{:x}", Sha256::digest(&raw))}}));
            }
            Ok(results)
        })?;
        for measurement in measurements {
            mismatch |= measurement["cosine"]["minimum"].as_f64().unwrap() < 0.9999;
            summary["programs"]
                .as_array_mut()
                .unwrap()
                .push(measurement);
        }
        save(summary)?;
    }
    ensure!(
        !mismatch,
        "late component diagnostic cosine below 0.9999; all measurements recorded"
    );
    Ok(())
}

#[test]
fn component_programs_match_natural_seams_without_private_api() {
    let model = crate::parakeet::tests::synthetic_model(true);
    for name in [
        "norm_feed_forward1",
        "norm_self_att",
        "norm_conv",
        "norm_feed_forward2",
        "norm_out",
        "feed_forward1",
        "attention",
        "convolution",
        "feed_forward2",
        "block",
    ] {
        let graph = component_graph(&model, 128, 23, name).unwrap();
        let (_, inputs, outputs) = graph.diagnostic_shapes();
        assert_eq!(
            inputs.len(),
            if ["attention", "convolution", "block"].contains(&name) {
                2
            } else {
                1
            }
        );
        assert_eq!(outputs, [crate::modernbert::shape(128, HIDDEN)]);
    }
    assert!(component_graph(&model, 128, 0, "block").is_err());
    assert_eq!(
        channel_major(&[0.0; 1], 128).unwrap_err().to_string(),
        "malformed full-rung seam input"
    );
}
