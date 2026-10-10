//! Accuracy audit for tight packing. Every row of a recorded code-search
//! export (the replay input) is embedded packed, once per rotary encoding
//! (nearest-rounded and constant-compatible), and compared with the same row
//! embedded alone. An encoding that fails any row reports no wall time.
use super::*;

type Ranges = Vec<std::ops::Range<usize>>;

fn bit_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

fn score(reference: &[Vec<f32>], outputs: &[Vec<f32>]) -> Value {
    assert_eq!(reference.len(), outputs.len());
    let mut min_cosine = 1.0f64;
    let mut maximum = 0.0f64;
    let mut identical = 0;
    let mut failed = Vec::new();
    for (index, (a, b)) in reference.iter().zip(outputs).enumerate() {
        let finite = a.len() == b.len() && a.iter().chain(b).all(|v| v.is_finite());
        let cos = if finite { cosine(a, b) } else { f64::NAN };
        if cos.is_finite() {
            min_cosine = min_cosine.min(cos);
        }
        if finite {
            maximum = maximum.max(max_abs(a, b));
        }
        identical += usize::from(bit_equal(a, b));
        if !cos.is_finite() || cos < 0.9999 {
            failed.push(index);
        }
    }
    json!({"rows": outputs.len(), "min_cosine": min_cosine, "max_abs": maximum, "bit_identical": identical, "failed_rows": failed, "passed": failed.is_empty()})
}

fn row_audit(
    chunks: &[Chunk],
    ranges: &Ranges,
    plans: &[tight::TightPlan],
    reference: &[Vec<f32>],
    outputs: &[Vec<f32>],
) -> Value {
    let mut records = vec![Value::Null; chunks.len()];
    let mut pass_index = 0;
    for (window_index, (range, plan)) in ranges.iter().zip(plans).enumerate() {
        assert!(plan.singles.is_empty(), "audit widths must fit this export");
        for indices in &plan.passes {
            let used: usize = indices
                .iter()
                .map(|&i| chunks[range.start + i].ids.len())
                .sum();
            let mut position = 0;
            for (slot, &i) in indices.iter().enumerate() {
                let index = range.start + i;
                let chunk = &chunks[index];
                let finite = reference[index].len() == outputs[index].len()
                    && reference[index]
                        .iter()
                        .chain(&outputs[index])
                        .all(|v| v.is_finite());
                let cos = if finite {
                    cosine(&reference[index], &outputs[index])
                } else {
                    f64::NAN
                };
                records[index] = json!({"seq": chunk.seq, "text_sha256": chunk.text_sha256, "tokens": chunk.ids.len(), "window_index": window_index, "pass_index": pass_index, "row_in_pass": slot, "position_in_pass": position, "last_token_position": position + chunk.ids.len() - 1, "pass_tokens": used, "cosine": cos.is_finite().then_some(cos), "max_abs": finite.then(|| max_abs(&reference[index], &outputs[index])), "bit_identical": bit_equal(&reference[index], &outputs[index]), "passed": cos.is_finite() && cos >= 0.9999});
                position += chunk.ids.len();
            }
            pass_index += 1;
        }
    }
    assert!(records.iter().all(|r| !r.is_null()));
    let worst = records
        .iter()
        .min_by(|a, b| {
            a["cosine"]
                .as_f64()
                .unwrap_or(-1.0)
                .total_cmp(&b["cosine"].as_f64().unwrap_or(-1.0))
        })
        .unwrap()
        .clone();
    json!({"summary": score(reference, outputs), "worst_row": worst, "rows": records})
}

fn fixture_audit(model: &Model, program: &tight::TightProgram) -> Value {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(format!(
            "../../bench/parity/fixtures/{SLUG}/{SLUG}.ref-v1.transformers-5.16.1.seed-0.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let cases: Vec<&Value> = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["input_ids"].as_array().unwrap().len() <= 256)
        .collect();
    let ids: Vec<Vec<u32>> = cases
        .iter()
        .map(|c| serde_json::from_value(c["input_ids"].clone()).unwrap())
        .collect();
    let reference: Vec<Vec<f32>> = ids.iter().map(|r| model.run(r).unwrap()).collect();
    let alone: Vec<Vec<f32>> = ids
        .iter()
        .map(|r| program.run(model, &[r]).unwrap().remove(0))
        .collect();
    let plan = tight::plan_tight(
        &ids.iter().map(Vec::len).collect::<Vec<_>>(),
        program.width(),
    )
    .unwrap();
    let mut mixed = vec![Vec::new(); ids.len()];
    for indices in plan.passes {
        let rows: Vec<&[u32]> = indices.iter().map(|&i| ids[i].as_slice()).collect();
        for (&i, vector) in indices.iter().zip(program.run(model, &rows).unwrap()) {
            mixed[i] = vector;
        }
    }
    json!({"cases": cases.iter().map(|c| c["id"].clone()).collect::<Vec<_>>(), "alone": score(&reference, &alone), "mixed": score(&reference, &mixed)})
}

fn isolation(model: &Model, program: &tight::TightProgram, chunks: &[Chunk]) -> Value {
    let short: Vec<&Chunk> = chunks
        .iter()
        .filter(|c| c.ids.len() <= 64)
        .take(3)
        .collect();
    let target = short[0].ids.as_slice();
    let a = program.run(model, &[target]).unwrap().remove(0);
    let b = program
        .run(model, &[target, &short[1].ids])
        .unwrap()
        .remove(0);
    let c = program
        .run(model, &[&short[2].ids, target])
        .unwrap()
        .pop()
        .unwrap();
    json!({"alone_vs_neighbours_max_abs": max_abs(&a, &b), "first_vs_last_max_abs": max_abs(&b, &c), "first_vs_last_cosine": cosine(&b, &c), "bit_identical_neighbours": bit_equal(&a, &b), "bit_identical_positions": bit_equal(&b, &c), "passed": bit_equal(&a, &b) && bit_equal(&b, &c)})
}

fn plans(chunks: &[Chunk], ranges: &Ranges, width: usize) -> Vec<tight::TightPlan> {
    ranges
        .iter()
        .map(|r| {
            tight::plan_tight(
                &chunks[r.clone()]
                    .iter()
                    .map(|c| c.ids.len())
                    .collect::<Vec<_>>(),
                width,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn row_audit_records_original_order_and_actual_positions_after_sorting() {
    let chunks: Vec<Chunk> = [100, 160, 56]
        .iter()
        .enumerate()
        .map(|(i, &n)| Chunk {
            seq: i as u64,
            text_sha256: "0".repeat(64),
            ids: vec![1; n],
        })
        .collect();
    let ranges: Ranges = std::iter::once(0..3).collect();
    let plans = plans(&chunks, &ranges, 256);
    let vectors = vec![vec![1., 0.]; 3];
    let audit = row_audit(&chunks, &ranges, &plans, &vectors, &vectors);
    assert_eq!(audit["rows"][0]["position_in_pass"], 0);
    assert_eq!(audit["rows"][1]["position_in_pass"], 0);
    assert_eq!(audit["rows"][2]["position_in_pass"], 160);
    assert_eq!(audit["rows"][2]["last_token_position"], 215);
    assert_eq!(audit["summary"]["passed"], true);
    let wrong = vec![vec![0., 1.], vec![1., 0.], vec![1., 0.]];
    let audit = row_audit(&chunks, &ranges, &plans, &vectors, &wrong);
    assert_eq!(audit["summary"]["failed_rows"], json!([0]));
    assert_eq!(audit["worst_row"]["seq"], 0);
    assert_eq!(audit["worst_row"]["max_abs"], 1.0);
}

/// Compare every row with standalone output under 64-row and 8-row planning,
/// plus reference fixtures and neighbour/position invariance. Require cosine
/// >= 0.9999 for all rows and exact isolation before timing an encoding/width.
#[test]
#[ignore = "Mac mini only: exhaustive nearest/constant-compatible rotary audit"]
fn tight_packing_audit() {
    crate::worker::test_private_api().unwrap();
    let env = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
    let input = env("ANE_MULTIROW_INPUT");
    let tokenizer = env("ANE_MULTIROW_TOKENIZER");
    let out = env("ANE_MULTIROW_OUT");
    let profile = Profile::select(&format!("{SLUG}.ane-direct-worker"), "embed").unwrap();
    let digest = profile.numeric["converted_package_digest"]
        .as_str()
        .unwrap()
        .to_owned();
    let chunks = tokenize_export(&input, &tokenizer, profile.n("eos_token_id") as u32);
    assert!(chunks.iter().all(|c| c.ids.len() <= 256));
    let meta: Value =
        serde_json::from_slice(&std::fs::read(format!("{input}.meta.json")).unwrap()).unwrap();
    let mut model = Model::load(
        profile,
        &std::path::PathBuf::from(env("ANE_TEST_PACKAGES")).join(format!("{SLUG}.safetensors")),
        &digest,
    )
    .unwrap();
    let mut report = json!({"export_sha256": sha(&std::fs::read(&input).unwrap()), "meta_sha256": sha(&std::fs::read(format!("{input}.meta.json")).unwrap()), "tokenizer_sha256": sha(&std::fs::read(&tokenizer).unwrap()), "package_digest": digest, "rows": chunks.len(), "batches": meta["batches"].as_array().unwrap().len(), "tokens": chunks.iter().map(|c| c.ids.len()).sum::<usize>(), "pid": std::process::id(), "started_unix_ms": unix_ms(), "source_commit": source_commit(), "release": !cfg!(debug_assertions), "audit": [], "cold_samples": [], "samples": [], "plans": [], "pass_cost": [], "setup": []});
    let write =
        |report: &Value| std::fs::write(&out, serde_json::to_vec_pretty(report).unwrap()).unwrap();
    for rung in [128, 256] {
        model.admit(rung, "tight-audit").unwrap();
    }
    report["single_fp32"] = fixture_gate(&model, &tokenizer);
    let reference: Vec<Vec<f32>> = chunks.iter().map(|c| model.run(&c.ids).unwrap()).collect();
    let ranges: Vec<Ranges> = [64, 8].map(|n| windows(&meta, chunks.len(), n)).into();
    write(&report);
    for width in [256, 384, 512] {
        let plan_sets: Vec<Vec<tight::TightPlan>> =
            ranges.iter().map(|r| plans(&chunks, r, width)).collect();
        for (j, window) in [64, 8].into_iter().enumerate() {
            let count: usize = plan_sets[j].iter().map(|p| p.passes.len()).sum();
            report["plans"].as_array_mut().unwrap().push(json!({"width": width, "window": window, "windows": ranges[j].len(), "passes": count, "fill_ratio": chunks.iter().map(|c| c.ids.len()).sum::<usize>() as f64 / (width * count) as f64}));
        }
        let mut qualified = Vec::new();
        for (arm, mode) in [
            ("nearest", tight::OperandMode::Runtime),
            (
                "constant-compatible",
                tight::OperandMode::RuntimeConstantCompatible,
            ),
        ] {
            let started = Instant::now();
            let program = model.compile_tight_mode(width, mode.clone());
            report["setup"].as_array_mut().unwrap().push(json!({"width": width, "arm": arm, "compile_ms": ms(started), "executables_held": 84, "compiled": program.is_ok(), "error": program.as_ref().err().map(|e| format!("{e:#}"))}));
            let program = match program {
                Ok(p) => p,
                Err(_) => {
                    write(&report);
                    continue;
                }
            };
            let fixture = fixture_audit(&model, &program);
            let isolated = isolation(&model, &program, &chunks);
            let mut passed = fixture["alone"]["passed"] == true
                && fixture["mixed"]["passed"] == true
                && isolated["passed"] == true;
            for j in 0..2 {
                let utc_start = unix_ms();
                let started = Instant::now();
                let outputs = execute_tight(&model, &program, &chunks, &ranges[j], &plan_sets[j]);
                let audit_ms = ms(started);
                let utc_end = unix_ms();
                let audit = row_audit(&chunks, &ranges[j], &plan_sets[j], &reference, &outputs);
                passed &= audit["summary"]["passed"] == true;
                println!(
                    "AUDIT width={width} arm={arm} window={} min_cos={} failures={}",
                    [64, 8][j],
                    audit["summary"]["min_cosine"],
                    audit["summary"]["failed_rows"].as_array().unwrap().len()
                );
                report["audit"].as_array_mut().unwrap().push(json!({"width": width, "arm": arm, "window": ([64, 8][j]), "started_unix_ms": utc_start, "finished_unix_ms": utc_end, "audit_ms_not_benchmark": audit_ms, "fixture": fixture, "isolation": isolated, "results": audit}));
                write(&report);
            }
            if passed {
                qualified.push((arm, mode));
            }
        }
        let timed_indices = plan_sets[0]
            .iter()
            .zip(&ranges[0])
            .flat_map(|(p, r)| {
                p.passes
                    .iter()
                    .map(move |indices| indices.iter().map(|&i| r.start + i).collect::<Vec<_>>())
            })
            .filter(|indices| {
                indices.len() >= 2
                    && indices
                        .iter()
                        .any(|&i| chunks[i].ids.len() != chunks[indices[0]].ids.len())
            })
            .max_by_key(|indices| indices.iter().map(|&i| chunks[i].ids.len()).sum::<usize>())
            .unwrap();
        let timed_rows: Vec<&[u32]> = timed_indices
            .iter()
            .map(|&i| chunks[i].ids.as_slice())
            .collect();
        let lengths: Vec<usize> = timed_rows.iter().map(|r| r.len()).collect();
        let expected: Vec<Vec<f32>> = timed_indices
            .iter()
            .map(|&i| reference[i].clone())
            .collect();
        let (_, base_cos, base_sin) = tight::runtime_operands(
            &lengths,
            width,
            model.profile.n("head_dim"),
            model.profile.f("rope_theta"),
        )
        .unwrap();
        let mut cos = base_cos.clone();
        let mut sin = base_sin.clone();
        let mut encode_ms = 0.0;
        for _ in 0..100 {
            cos.clone_from(&base_cos);
            sin.clone_from(&base_sin);
            let started = Instant::now();
            tight::constant_compatible_coefficients(&mut cos);
            tight::constant_compatible_coefficients(&mut sin);
            encode_ms += ms(started);
            std::hint::black_box((&cos, &sin));
        }
        // Only when every row passed: time the constant-mask/rotary,
        // runtime-mask and constant-compatible runtime variants. Evict the
        // single-row 128 and 256 programs first, because three 28-layer
        // variants need 84 compiled executables of the worker's ~100 budget.
        evict_all(&mut model);
        let mut modes = vec![
            ("constants", tight::OperandMode::Constants(lengths.clone())),
            (
                "runtime-mask",
                tight::OperandMode::MaskOnly(lengths.clone()),
            ),
        ];
        if qualified
            .iter()
            .any(|(arm, _)| *arm == "constant-compatible")
        {
            modes.push((
                "constant-compatible",
                tight::OperandMode::RuntimeConstantCompatible,
            ));
        }
        let mut controls = Vec::new();
        let mut programs = Vec::new();
        for (name, mode) in modes {
            let program = model.compile_tight_mode(width, mode).unwrap();
            let outputs = program.run(&model, &timed_rows).unwrap();
            let gate = score(&expected, &outputs);
            assert_eq!(gate["passed"], true, "control {name} width {width}: {gate}");
            controls.push(json!({"mode": name, "parity": gate, "samples": []}));
            programs.push(program);
        }
        for repeat in 0..9 {
            let mut order: Vec<usize> = (0..programs.len()).collect();
            if repeat % 2 == 1 {
                order.reverse();
            }
            for i in order {
                let utc_start = unix_ms();
                let started = Instant::now();
                let outputs = programs[i].run(&model, &timed_rows).unwrap();
                let wall = ms(started);
                let utc_end = unix_ms();
                assert_eq!(score(&expected, &outputs)["passed"], true);
                controls[i]["samples"].as_array_mut().unwrap().push(json!({"repeat": repeat, "wall_ms": wall, "started_unix_ms": utc_start, "finished_unix_ms": utc_end}));
            }
        }
        drop(programs);
        report["pass_cost"].as_array_mut().unwrap().push(json!({"width": width, "segment_lengths": lengths, "encode_decode_ms": encode_ms / 100.0, "encode_decode_samples": 100, "variants": controls}));
        for rung in [128, 256] {
            model.admit(rung, "tight-audit").unwrap();
        }
        write(&report);
        for (arm, mode) in qualified {
            let program = model.compile_tight_mode(width, mode).unwrap();
            let load = load_average();
            let utc_start = unix_ms();
            let started = Instant::now();
            let warm_single: Vec<Vec<f32>> =
                chunks.iter().map(|c| model.run(&c.ids).unwrap()).collect();
            let wall = ms(started);
            let utc_end = unix_ms();
            assert_eq!(score(&reference, &warm_single)["passed"], true);
            report["cold_samples"].as_array_mut().unwrap().push(json!({"width": width, "arm": arm, "method": "single", "window": 0, "cold_shapes": ["single-128", "single-256"], "wall_ms": wall, "started_unix_ms": utc_start, "finished_unix_ms": utc_end, "load_before": load, "load_after": load_average()}));
            for j in 0..2 {
                let load = load_average();
                let utc_start = unix_ms();
                let started = Instant::now();
                let warm = execute_tight(&model, &program, &chunks, &ranges[j], &plan_sets[j]);
                let wall = ms(started);
                let utc_end = unix_ms();
                assert_eq!(score(&reference, &warm)["passed"], true);
                if j == 0 {
                    report["cold_samples"].as_array_mut().unwrap().push(json!({"width": width, "arm": arm, "method": "tight", "window": 64, "cold_shapes": [format!("tight-{width}-{arm}")], "wall_ms": wall, "started_unix_ms": utc_start, "finished_unix_ms": utc_end, "load_before": load, "load_after": load_average()}));
                }
            }
            write(&report);
            for repeat in 0..3 {
                let order = if repeat % 2 == 0 {
                    [0, 1, 2]
                } else {
                    [2, 1, 0]
                };
                for path in order {
                    let load = load_average();
                    let utc_start = unix_ms();
                    let started = Instant::now();
                    let outputs = if path == 0 {
                        chunks.iter().map(|c| model.run(&c.ids).unwrap()).collect()
                    } else {
                        let j = path - 1;
                        let live_plans = plans(&chunks, &ranges[j], width);
                        execute_tight(&model, &program, &chunks, &ranges[j], &live_plans)
                    };
                    let wall = ms(started);
                    let utc_end = unix_ms();
                    let gate = score(&reference, &outputs);
                    assert_eq!(gate["passed"], true);
                    report["samples"].as_array_mut().unwrap().push(json!({"width": width, "arm": arm, "method": if path == 0 { "single" } else { "tight" }, "window": if path == 0 { 0 } else { [64, 8][path - 1] }, "repeat": repeat, "wall_ms": wall, "rows_per_second": chunks.len() as f64 * 1000.0 / wall, "started_unix_ms": utc_start, "finished_unix_ms": utc_end, "load_before": load, "load_after": load_average(), "parity": gate}));
                    write(&report);
                    println!("TIGHT-AUDIT width={width} arm={arm} repeat={repeat} path={path} wall_s={:.3}", wall / 1000.0);
                }
            }
        }
    }
    evict_all(&mut model);
    write(&report);
}
