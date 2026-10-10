//! Whole-export measurements with phase-separated executable residency.
//! No serving entry point calls this ignored hardware harness.
use super::hardware::{
    compose, cosine, load_average, max_abs, ms, sha, tokenize_export, Chunk, SLUG,
};
use super::*;
use crate::backend::{rung, Profile};
use serde_json::{json, Value};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Record each sample's UTC start/end to detect overlapping Neural Engine runs.
// Instant measures inference duration independently of wall-clock adjustments.
fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .try_into()
        .unwrap()
}

fn source_commit() -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn windows(meta: &Value, total: usize, limit: usize) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut next = 0;
    for batch in meta["batches"].as_array().unwrap() {
        let first = batch["first_chunk_seq"].as_u64().unwrap() as usize;
        let count = batch["chunk_count"].as_u64().unwrap() as usize;
        assert_eq!(first, next, "export batches must cover rows consecutively");
        for start in (first..first + count).step_by(limit) {
            ranges.push(start..(start + limit).min(first + count));
        }
        next += count;
    }
    assert_eq!(next, total);
    ranges
}

fn parity(reference: &[Vec<f32>], outputs: &[Vec<f32>]) -> Value {
    assert_eq!(outputs.len(), reference.len());
    let mut min_cos = 1.0f64;
    let mut maximum = 0.0f64;
    let mut identical = 0;
    for (index, (a, b)) in reference.iter().zip(outputs).enumerate() {
        assert_eq!(a.len(), b.len(), "row {index} dimensions");
        let cos = cosine(a, b);
        assert!(cos.is_finite() && cos >= 0.9999, "row {index} cosine {cos}");
        min_cos = min_cos.min(cos);
        maximum = maximum.max(max_abs(a, b));
        identical += usize::from(a == b);
    }
    json!({"rows": outputs.len(), "min_cosine": min_cos, "max_abs": maximum, "identical": identical})
}

fn evict_all(model: &mut Model) {
    let rungs: Vec<usize> = model.resident.keys().copied().collect();
    for rung in rungs {
        model.evict(rung);
    }
}

fn fixture_gate(model: &Model, tokenizer_path: &str) -> Value {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(format!(
            "../../bench/parity/fixtures/{SLUG}/{SLUG}.ref-v1.transformers-5.16.1.seed-0.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap();
    let mut cases = Vec::new();
    let mut matching = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
        if let Some(text) = case["text"].as_str() {
            assert_eq!(
                compose(&tokenizer, text, model.profile.n("eos_token_id") as u32),
                ids
            );
            matching += 1;
        }
        if !model.resident.contains_key(&rung(ids.len()).unwrap()) {
            continue;
        }
        let expected: Vec<f32> = serde_json::from_value(case["output"].clone()).unwrap();
        let output = model.run(&ids).unwrap();
        let cos = cosine(&expected, &output);
        assert!(cos.is_finite() && cos >= 0.999, "fp32 {} {cos}", case["id"]);
        cases.push(json!({"case": case["id"], "tokens": ids.len(), "cosine": cos, "max_abs": max_abs(&expected, &output)}));
    }
    json!({"composition_matches": matching, "cases": cases})
}

fn execute_fixed(
    model: &Model,
    programs: &[MultiRowProgram],
    chunks: &[Chunk],
    plans: &[Vec<Pass>],
    ranges: &[std::ops::Range<usize>],
) -> Vec<Vec<f32>> {
    let mut outputs = vec![Vec::new(); chunks.len()];
    for (range, passes) in ranges.iter().zip(plans) {
        for pass in passes {
            match pass {
                Pass::Single(index) => {
                    outputs[range.start + index] = model
                        .run_at_rung(&chunks[range.start + index].ids, 256)
                        .unwrap()
                }
                Pass::MultiRow { program, rows } => {
                    let tokens: Vec<&[u32]> = rows
                        .iter()
                        .map(|i| chunks[range.start + i].ids.as_slice())
                        .collect();
                    let vectors = programs[*program].run(model, &tokens).unwrap();
                    for (index, vector) in rows.iter().zip(vectors) {
                        outputs[range.start + index] = vector;
                    }
                }
            }
        }
    }
    outputs
}

#[test]
fn windows_respect_export_batches_and_do_not_merge_eight_row_tails() {
    let meta = json!({"batches": [{"first_chunk_seq": 0, "chunk_count": 10}, {"first_chunk_seq": 10, "chunk_count": 9}]});
    assert_eq!(windows(&meta, 19, 64), vec![0..10, 10..19]);
    assert_eq!(windows(&meta, 19, 8), vec![0..8, 8..10, 10..18, 18..19]);
}

/// Requires ANE_MULTIROW_INPUT (export), ANE_MULTIROW_TOKENIZER,
/// ANE_TEST_PACKAGES (converted package directory), and ANE_MULTIROW_OUT.
/// Residency changes happen outside the inference clock and are reported.
#[test]
#[ignore = "Mac mini only: full export, alternating residency phases"]
fn whole_export_planner() {
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
    let meta: Value =
        serde_json::from_slice(&std::fs::read(format!("{input}.meta.json")).unwrap()).unwrap();
    let mut model = Model::load(
        profile,
        &std::path::PathBuf::from(env("ANE_TEST_PACKAGES")).join(format!("{SLUG}.safetensors")),
        &digest,
    )
    .unwrap();
    let confirm = std::env::var_os("ANE_MULTIROW_CONFIRM").is_some();
    let shapes = [(8, 32), (4, 64), (2, 128), (2, 64)]
        .map(|(rows, width)| MultiRowShape::new(RowLayout::WidthFolded, rows, width).unwrap());
    // Prior measurements found 256 columns most efficient for full passes.
    // Confirmation also tries two 64-token slots (128 columns) for windows
    // with only one or two short rows, where four slots would waste work.
    let configs = if confirm {
        vec![
            ("ceiling-112", vec![0, 1, 2]),
            ("deployable-32-128", vec![0, 2]),
            ("deployable-64-128", vec![1, 2]),
            ("deployable-small64-128", vec![3, 2]),
        ]
    } else {
        vec![
            ("ceiling-112", vec![0, 1, 2]),
            ("deployable-32-64", vec![0, 1]),
            ("deployable-32-128", vec![0, 2]),
            ("deployable-64-128", vec![1, 2]),
        ]
    };
    let mut report = json!({"export_sha256": sha(&std::fs::read(&input).unwrap()), "meta_sha256": sha(&std::fs::read(format!("{input}.meta.json")).unwrap()), "tokenizer_sha256": sha(&std::fs::read(&tokenizer).unwrap()), "package_digest": digest, "rows": chunks.len(), "batches": meta["batches"].as_array().unwrap().len(), "tokens": chunks.iter().map(|c| c.ids.len()).sum::<usize>(), "rows_above_256": chunks.iter().filter(|c| c.ids.len() > 256).count(), "pid": std::process::id(), "started_unix_ms": unix_ms(), "source_commit": source_commit(), "release": !cfg!(debug_assertions), "full_export_warm": confirm, "cold_samples": [], "samples": [], "setup": [], "plans": []});
    let write =
        |report: &Value| std::fs::write(&out, serde_json::to_vec_pretty(report).unwrap()).unwrap();
    // This export has no rows above 256. Refuse other exports because their
    // long rows require timed eviction/recompile support, not measured here.
    assert!(
        chunks.iter().all(|c| c.ids.len() <= 256),
        "long rows require timed eviction/recompile support"
    );
    let mut reference = Vec::new();
    if confirm {
        let started = Instant::now();
        for width in [128, 256] {
            model.admit(width, "planner-preflight").unwrap();
        }
        report["single_fp32"] = fixture_gate(&model, &tokenizer);
        report["wider_fallback_gate"] = wider_fallback_gate(&model, &chunks);
        reference = chunks.iter().map(|c| model.run(&c.ids).unwrap()).collect();
        report["preflight_ms"] = json!(ms(started));
        evict_all(&mut model);
        write(&report);
    }
    for repeat in 0..3 {
        let mut order: Vec<usize> = (0..=configs.len()).collect();
        if repeat % 2 == 1 {
            order.reverse();
        }
        for phase in order {
            evict_all(&mut model);
            let setup_utc = unix_ms();
            let setup_start = Instant::now();
            if phase == 0 {
                for width in [128, 256] {
                    model.admit(width, "whole-export-planner").unwrap();
                }
                report["setup"].as_array_mut().unwrap().push(json!({"repeat": repeat, "method": "single", "ms": ms(setup_start), "started_unix_ms": setup_utc, "finished_unix_ms": unix_ms(), "executables": 56}));
                if repeat == 0 && !confirm {
                    report["single_fp32"] = fixture_gate(&model, &tokenizer);
                    report["wider_fallback_gate"] = wider_fallback_gate(&model, &chunks);
                }
                if confirm {
                    let load = load_average();
                    let utc_start = unix_ms();
                    let start = Instant::now();
                    let outputs: Vec<Vec<f32>> =
                        chunks.iter().map(|c| model.run(&c.ids).unwrap()).collect();
                    report["cold_samples"].as_array_mut().unwrap().push(json!({"repeat": repeat, "method": "single", "window": 0, "cold_shapes": ["single-128", "single-256"], "wall_ms": ms(start), "load_before": load, "started_unix_ms": utc_start, "finished_unix_ms": unix_ms(), "load_after": load_average(), "parity": parity(&reference, &outputs)}));
                } else {
                    model.run(&chunks[0].ids).unwrap();
                }
                let load = load_average();
                let utc_start = unix_ms();
                let start = Instant::now();
                let outputs: Vec<Vec<f32>> =
                    chunks.iter().map(|c| model.run(&c.ids).unwrap()).collect();
                let wall = ms(start);
                if reference.is_empty() {
                    reference = outputs.clone();
                }
                report["samples"].as_array_mut().unwrap().push(json!({"repeat": repeat, "method": "single", "window": 0, "wall_ms": wall, "rows_per_second": chunks.len() as f64 * 1000.0 / wall, "load_before": load, "started_unix_ms": utc_start, "finished_unix_ms": unix_ms(), "load_after": load_average(), "parity": parity(&reference, &outputs)}));
            } else {
                let (name, indices) = &configs[phase - 1];
                model.admit(256, "whole-export-planner").unwrap();
                let selected: Vec<MultiRowShape> = indices.iter().map(|&i| shapes[i]).collect();
                let programs: Vec<MultiRowProgram> = selected
                    .iter()
                    .map(|&s| model.compile_multirow(s).unwrap())
                    .collect();
                report["setup"].as_array_mut().unwrap().push(json!({"repeat": repeat, "method": name, "ms": ms(setup_start), "started_unix_ms": setup_utc, "finished_unix_ms": unix_ms(), "executables": 28 * (1 + programs.len())}));
                for program in programs.iter().filter(|_| !confirm) {
                    let indices: Vec<usize> = chunks
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| c.ids.len() <= program.shape().width)
                        .take(program.shape().rows)
                        .map(|(i, _)| i)
                        .collect();
                    let tokens: Vec<&[u32]> =
                        indices.iter().map(|&i| chunks[i].ids.as_slice()).collect();
                    let outputs = program.run(&model, &tokens).unwrap();
                    let expected: Vec<Vec<f32>> =
                        indices.iter().map(|&i| reference[i].clone()).collect();
                    parity(&expected, &outputs);
                }
                let window_order = if repeat % 2 == 0 { [64, 8] } else { [8, 64] };
                for window in window_order {
                    let ranges = windows(&meta, chunks.len(), window);
                    let plan_start = Instant::now();
                    let plans: Vec<Vec<Pass>> = ranges
                        .iter()
                        .map(|r| {
                            plan_passes(
                                &chunks[r.clone()]
                                    .iter()
                                    .map(|c| c.ids.len())
                                    .collect::<Vec<_>>(),
                                &selected,
                            )
                        })
                        .collect();
                    let plan_ms = ms(plan_start);
                    let single_passes = plans
                        .iter()
                        .flatten()
                        .filter(|p| matches!(p, Pass::Single(_)))
                        .count();
                    let multi_counts: Vec<Value> = selected.iter().enumerate().map(|(program, shape)| json!({"shape": shape.label(), "passes": plans.iter().flatten().filter(|p| matches!(p, Pass::MultiRow {program: q, ..} if *q == program)).count()})).collect();
                    if repeat == 0 {
                        report["plans"].as_array_mut().unwrap().push(json!({"method": name, "window": window, "windows": ranges.len(), "single_passes": single_passes, "multi_counts": multi_counts, "evictions_in_wall": 0, "recompiles_in_wall": 0}));
                    }
                    if confirm && window == window_order[0] {
                        let load = load_average();
                        let utc_start = unix_ms();
                        let start = Instant::now();
                        let outputs = execute_fixed(&model, &programs, &chunks, &plans, &ranges);
                        let wall = ms(start) + plan_ms;
                        let mut cold_shapes: Vec<String> =
                            programs.iter().map(|p| p.shape().label()).collect();
                        cold_shapes.push("single-256".to_owned());
                        report["cold_samples"].as_array_mut().unwrap().push(json!({"repeat": repeat, "method": name, "window": window, "cold_shapes": cold_shapes, "wall_ms": wall, "load_before": load, "started_unix_ms": utc_start, "finished_unix_ms": unix_ms(), "load_after": load_average(), "parity": parity(&reference, &outputs)}));
                        write(&report);
                    }
                    let load = load_average();
                    let utc_start = unix_ms();
                    let start = Instant::now();
                    let outputs = execute_fixed(&model, &programs, &chunks, &plans, &ranges);
                    let wall = ms(start) + plan_ms;
                    report["samples"].as_array_mut().unwrap().push(json!({"repeat": repeat, "method": name, "window": window, "wall_ms": wall, "planning_ms": plan_ms, "rows_per_second": chunks.len() as f64 * 1000.0 / wall, "load_before": load, "started_unix_ms": utc_start, "finished_unix_ms": unix_ms(), "load_after": load_average(), "parity": parity(&reference, &outputs)}));
                    write(&report);
                    println!(
                        "PLANNER repeat={repeat} method={name} window={window} wall_s={:.3}",
                        wall / 1000.0
                    );
                }
                drop(programs);
            }
            write(&report);
        }
    }
    evict_all(&mut model);
}

fn execute_tight(
    model: &Model,
    program: &tight::TightProgram,
    chunks: &[Chunk],
    ranges: &[std::ops::Range<usize>],
    plans: &[tight::TightPlan],
) -> Vec<Vec<f32>> {
    let mut outputs = vec![Vec::new(); chunks.len()];
    for (range, plan) in ranges.iter().zip(plans) {
        for &i in &plan.singles {
            outputs[range.start + i] = model.run(&chunks[range.start + i].ids).unwrap();
        }
        for indices in &plan.passes {
            let rows: Vec<&[u32]> = indices
                .iter()
                .map(|&i| chunks[range.start + i].ids.as_slice())
                .collect();
            let vectors = program.run(model, &rows).unwrap();
            for (&i, vector) in indices.iter().zip(vectors) {
                outputs[range.start + i] = vector;
            }
        }
    }
    outputs
}

/// Requires ANE_MULTIROW_INPUT, ANE_MULTIROW_TOKENIZER, ANE_TEST_PACKAGES,
/// and ANE_MULTIROW_OUT. Measures
/// constant, runtime-mask-only and fully runtime graphs at identical geometry
/// before the whole-export fully-runtime replay. No segment is a graph constant
/// in the candidate replay.
#[test]
#[ignore = "Mac mini only: runtime-input packing parity and whole-export timing"]
fn tight_packing_experiment() {
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
    let meta: Value =
        serde_json::from_slice(&std::fs::read(format!("{input}.meta.json")).unwrap()).unwrap();
    let mut model = Model::load(
        profile,
        &std::path::PathBuf::from(env("ANE_TEST_PACKAGES")).join(format!("{SLUG}.safetensors")),
        &digest,
    )
    .unwrap();
    let mut report = json!({"export_sha256": sha(&std::fs::read(&input).unwrap()), "meta_sha256": sha(&std::fs::read(format!("{input}.meta.json")).unwrap()), "tokenizer_sha256": sha(&std::fs::read(&tokenizer).unwrap()), "package_digest": digest, "rows": chunks.len(), "batches": meta["batches"].as_array().unwrap().len(), "tokens": chunks.iter().map(|c| c.ids.len()).sum::<usize>(), "pid": std::process::id(), "started_unix_ms": unix_ms(), "source_commit": source_commit(), "release": !cfg!(debug_assertions), "cold_samples": [], "setup": [], "pass_cost": [], "plans": [], "samples": [], "correctness": []});
    let write = |r: &Value| std::fs::write(&out, serde_json::to_vec_pretty(r).unwrap()).unwrap();
    assert!(chunks.iter().all(|c| c.ids.len() <= 256));
    for width in [128, 256] {
        let start = Instant::now();
        model.admit(width, "tight-packing-experiment").unwrap();
        report["setup"]
            .as_array_mut()
            .unwrap()
            .push(json!({"single_width": width, "ms": ms(start)}));
    }
    report["single_fp32"] = fixture_gate(&model, &tokenizer);
    let reference: Vec<Vec<f32>> = chunks.iter().map(|c| model.run(&c.ids).unwrap()).collect();
    write(&report);
    let ranges: Vec<_> = [64, 8]
        .map(|window| windows(&meta, chunks.len(), window))
        .into();
    for width in [256, 384, 512] {
        let plans: Vec<Vec<tight::TightPlan>> = ranges
            .iter()
            .map(|ranges| {
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
            })
            .collect();
        for (j, window) in [64, 8].into_iter().enumerate() {
            let pass_count: usize = plans[j].iter().map(|p| p.passes.len()).sum();
            let tokens: usize = chunks.iter().map(|c| c.ids.len()).sum();
            report["plans"].as_array_mut().unwrap().push(json!({"width": width, "window": window, "passes": pass_count, "fill_ratio": tokens as f64 / (pass_count * width) as f64, "windows": ranges[j].len(), "single_passes": plans[j].iter().map(|p| p.singles.len()).sum::<usize>()}));
        }
        write(&report);
        // Select the real mixed pass with the most occupied columns. Every
        // variant sees the same rows, isolating runtime mask/RoPE input cost.
        let timed_indices = plans[0]
            .iter()
            .zip(&ranges[0])
            .flat_map(|(p, r)| {
                p.passes
                    .iter()
                    .map(move |indices| indices.iter().map(|&i| r.start + i).collect::<Vec<_>>())
            })
            .filter(|indices| indices.len() >= 2)
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
        let mut costs = Vec::new();
        // Evict both single-row programs first: three 28-layer variants need 84
        // compiled executables, which only fits the worker's budget of about
        // 100 once nothing else is resident. Alternate the variants' timing
        // order, then restore the baseline programs.
        evict_all(&mut model);
        let mut variants = Vec::new();
        for (name, mode) in [
            ("constants", tight::OperandMode::Constants(lengths.clone())),
            (
                "runtime-mask",
                tight::OperandMode::MaskOnly(lengths.clone()),
            ),
            ("runtime-mask-rope", tight::OperandMode::Runtime),
        ] {
            let start = Instant::now();
            let program = model.compile_tight_mode(width, mode).unwrap();
            report["setup"].as_array_mut().unwrap().push(json!({"width": width, "mode": name, "ms": ms(start), "executables_held": (variants.len() + 1) * program.executable_count()}));
            let output = program.run(&model, &timed_rows).unwrap();
            let gate = parity(&expected, &output);
            costs.push(json!({"mode": name, "samples": [], "parity": gate}));
            variants.push(program);
        }
        for repeat in 0..9 {
            let order = if repeat % 2 == 0 {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            };
            for index in order {
                let load = load_average();
                let utc_start = unix_ms();
                let start = Instant::now();
                let output = variants[index].run(&model, &timed_rows).unwrap();
                let wall = ms(start);
                parity(&expected, &output);
                costs[index]["samples"].as_array_mut().unwrap().push(json!({"repeat": repeat, "wall_ms": wall, "load_before": load, "started_unix_ms": utc_start, "finished_unix_ms": unix_ms(), "load_after": load_average()}));
            }
        }
        drop(variants);
        let restore_start = Instant::now();
        for rung in [128, 256] {
            model.admit(rung, "tight-packing-experiment").unwrap();
        }
        report["setup"].as_array_mut().unwrap().push(json!({"width": width, "mode": "restore-baseline", "ms": ms(restore_start), "executables_held": 56}));
        let host_start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(
                tight::runtime_operands(
                    &lengths,
                    width,
                    model.profile.n("head_dim"),
                    model.profile.f("rope_theta"),
                )
                .unwrap(),
            );
        }
        let host_operands_ms = ms(host_start) / 100.0;
        report["pass_cost"].as_array_mut().unwrap().push(json!({"host_operands_ms": host_operands_ms,"width": width, "segment_lengths": lengths, "rows": timed_indices.iter().map(|&i| json!({"seq": chunks[i].seq, "text_sha256": chunks[i].text_sha256})).collect::<Vec<_>>(), "fill_ratio": lengths.iter().sum::<usize>() as f64 / width as f64, "variants": costs}));
        write(&report);
        let start = Instant::now();
        let program = model
            .compile_tight_mode(width, tight::OperandMode::Runtime)
            .unwrap();
        report["setup"].as_array_mut().unwrap().push(json!({"width": width, "mode": "replay-runtime", "ms": ms(start), "executables_held": 56 + program.executable_count()}));
        assert_eq!(program.width(), width);
        let fixture = tight_fixture_gate(&model, &program);
        report["correctness"]
            .as_array_mut()
            .unwrap()
            .push(json!({"width": width, "fixture": fixture}));
        // Vary neighbouring content and move a real row from first to last.
        // Choose short neighbours so the target fits at every tested width.
        let short: Vec<usize> = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.ids.len() <= 64)
            .take(3)
            .map(|(i, _)| i)
            .collect();
        let target = chunks[short[0]].ids.as_slice();
        let a = program.run(&model, &[target]).unwrap().remove(0);
        let b = program
            .run(&model, &[target, &chunks[short[1]].ids])
            .unwrap()
            .remove(0);
        let c = program
            .run(&model, &[&chunks[short[2]].ids, target])
            .unwrap()
            .pop()
            .unwrap();
        let isolation = json!({"alone_vs_neighbours_max_abs": max_abs(&a, &b), "first_vs_last_max_abs": max_abs(&b, &c), "first_vs_last_cosine": cosine(&b, &c)});
        report["isolation"] = isolation.clone();
        write(&report);
        assert_eq!(a, b, "cross-row leakage at {width}");
        assert_eq!(b, c, "first/last position differs at {width}");
        for j in 0..2 {
            let utc_start = unix_ms();
            let started = Instant::now();
            let outputs = execute_tight(&model, &program, &chunks, &ranges[j], &plans[j]);
            let wall = ms(started);
            if j == 0 {
                report["cold_samples"].as_array_mut().unwrap().push(json!({"width": width, "method": "tight", "window": 64, "cold_shapes": [format!("tight-{width}"), "single-128", "single-256"], "short_fixture_warmup": true, "started_unix_ms": utc_start, "finished_unix_ms": unix_ms(), "wall_ms": wall, "parity": parity(&reference, &outputs)}));
            }
            report["correctness"].as_array_mut().unwrap().push(json!({"width": width, "window": ([64, 8][j]), "parity": parity(&reference, &outputs), "isolation": isolation}));
        }
        write(&report);
        for repeat in 0..3 {
            let mut order = vec![0, 1, 2];
            if repeat % 2 == 1 {
                order.reverse();
            }
            for path in order {
                let load = load_average();
                let utc_start = unix_ms();
                let start = Instant::now();
                let (outputs, planning_ms) = if path == 0 {
                    (
                        chunks.iter().map(|c| model.run(&c.ids).unwrap()).collect(),
                        0.0,
                    )
                } else {
                    let j = path - 1;
                    let plan_start = Instant::now();
                    let live_plans: Vec<tight::TightPlan> = ranges[j]
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
                        .collect();
                    let planning_ms = ms(plan_start);
                    (
                        execute_tight(&model, &program, &chunks, &ranges[j], &live_plans),
                        planning_ms,
                    )
                };
                let wall = ms(start);
                report["samples"].as_array_mut().unwrap().push(json!({"width": width, "repeat": repeat, "planning_ms": planning_ms, "method": if path == 0 { "single" } else { "tight" }, "window": if path == 0 { 0 } else { [64, 8][path - 1] }, "wall_ms": wall, "rows_per_second": chunks.len() as f64 * 1000.0 / wall, "load_before": load, "started_unix_ms": utc_start, "finished_unix_ms": unix_ms(), "load_after": load_average(), "parity": parity(&reference, &outputs)}));
                write(&report);
                println!(
                    "TIGHT width={width} repeat={repeat} path={path} wall_s={:.3}",
                    wall / 1000.0
                );
            }
        }
    }
    evict_all(&mut model);
}

fn tight_fixture_gate(model: &Model, program: &tight::TightProgram) -> Value {
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
        .filter(|case| {
            let n = case["input_ids"].as_array().unwrap().len();
            n <= program.width() && model.resident.contains_key(&rung(n).unwrap())
        })
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
    assert!(plan.singles.is_empty());
    let mut outputs = vec![Vec::new(); ids.len()];
    for indices in &plan.passes {
        let tokens: Vec<&[u32]> = indices.iter().map(|&i| ids[i].as_slice()).collect();
        for (&i, vector) in indices.iter().zip(program.run(model, &tokens).unwrap()) {
            outputs[i] = vector;
        }
    }
    json!({"cases": cases.iter().map(|c| c["id"].clone()).collect::<Vec<_>>(), "alone_vs_single": parity(&reference, &alone), "mixed_vs_single": parity(&reference, &outputs)})
}

fn wider_fallback_gate(model: &Model, chunks: &[Chunk]) -> Value {
    let rows: Vec<&Chunk> = chunks
        .iter()
        .filter(|c| (65..=128).contains(&c.ids.len()))
        .take(10)
        .collect();
    assert_eq!(rows.len(), 10);
    let mut cases = Vec::new();
    for c in rows {
        let start = Instant::now();
        let narrow = model.run(&c.ids).unwrap();
        let narrow_ms = ms(start);
        let start = Instant::now();
        let wide = model.run_at_rung(&c.ids, 256).unwrap();
        let wide_ms = ms(start);
        let cos = cosine(&narrow, &wide);
        assert!(
            cos.is_finite() && cos >= 0.9999,
            "wider rung changed row {} cosine {cos}",
            c.seq
        );
        cases.push(json!({"seq": c.seq, "text_sha256": c.text_sha256, "tokens": c.ids.len(), "cosine": cos, "max_abs": max_abs(&narrow, &wide), "identical": narrow == wide, "rung128_ms": narrow_ms, "rung256_ms": wide_ms}));
    }
    json!({"cases": cases})
}

#[path = "tight_audit.rs"]
mod audit;
