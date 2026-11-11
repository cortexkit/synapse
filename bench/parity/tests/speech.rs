use synapse_parity::speech::*;

#[test]
fn speech_addition_preserves_all_sixteen_legacy_profile_digests_and_shared_blocks() {
    let frozen: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/legacy-models.json")).unwrap();
    let manifest =
        synapse_parity::manifest::Manifest::load(&synapse_parity::parity_dir().join("models.json"))
            .unwrap();
    let current = serde_json::to_value(&manifest).unwrap();
    for key in ["schema", "admission", "converter"] {
        assert_eq!(current[key], frozen[key], "{key}");
    }
    let old = frozen["digests"]["profiles"].as_object().unwrap();
    assert_eq!(old.len(), 16);
    for (id, digest) in old {
        assert_eq!(
            manifest.profile_digest(id).unwrap(),
            digest.as_str().unwrap(),
            "{id}"
        );
        assert_eq!(current["digests"]["profiles"][id], *digest, "{id}");
        assert_eq!(current["profiles"][id], frozen["profiles"][id], "{id}");
    }
    for id in frozen["models"].as_object().unwrap().keys() {
        assert_eq!(current["models"][id], frozen["models"][id], "{id}");
    }
}

#[test]
fn speech_tensor_inventory_covers_every_pinned_tensor_and_integer_counter() {
    let manifest =
        synapse_parity::manifest::Manifest::load(&synapse_parity::parity_dir().join("models.json"))
            .unwrap();
    let model = manifest.model(MODEL).unwrap();
    assert!(model.grammar.is_none() && model.output.is_none());
    let header = synapse_parity::safetensors::parse_header_json(
        &std::fs::read(
            synapse_parity::parity_dir()
                .join("checkpoints")
                .join(MODEL)
                .join("tensor-index.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let expected = expected_tensors(&model.architecture).unwrap();
    assert_eq!(expected.len(), 723);
    assert_eq!(header.tensors.len(), expected.len());
    for (name, shape) in &expected {
        assert_eq!(&header.tensors[name].shape, shape, "{name}");
    }
    let counters: Vec<_> = header
        .tensors
        .iter()
        .filter(|(_, info)| info.dtype == synapse_parity::safetensors::StDType::I64)
        .collect();
    assert_eq!(counters.len(), 24);
    assert!(counters.iter().all(
        |(name, info)| name.ends_with(".conv.norm.num_batches_tracked") && info.shape.is_empty()
    ));
}

#[test]
fn speech_fixture_counts_traces_and_committed_tensor_seals_agree() {
    let root = synapse_parity::parity_dir();
    let manifest = synapse_parity::manifest::Manifest::load(&root.join("models.json")).unwrap();
    let fixtures = load_transcribe_fixtures(&root, &manifest).unwrap();
    assert_eq!(fixtures.iter().filter(|c| c.split.is_some()).count(), 1200);
    assert_eq!(
        fixtures
            .iter()
            .filter(|c| c.id.starts_with("ladder-"))
            .count(),
        22
    );
    let document: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("fixtures/parakeet-tdt-0.6b-v3/reference.json")).unwrap(),
    )
    .unwrap();
    let mut corpus_tensors = 0;
    for case in &fixtures {
        let raw = document["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == case.id)
            .unwrap();
        let (actual_mel, valid_mel, valid_encoder) = frame_counts(case.sample_count);
        assert_eq!(raw["mel_frames"], actual_mel, "{}", case.id);
        assert_eq!(raw["valid_mel_frames"], valid_mel, "{}", case.id);
        assert_eq!(case.valid_encoder_frames, valid_encoder, "{}", case.id);
        let trace = decode_trace(&case.trace.read(&root, &root).unwrap()).unwrap();
        let triples: Vec<Triple> = serde_json::from_value(raw["triples"].clone()).unwrap();
        assert!(!grade_trace(&trace, &trace, &triples, valid_encoder, 0.001).unwrap());
        assert_eq!(raw["no_speech"], triples.is_empty(), "{}", case.id);
        if raw.get("log_mel").is_some() {
            let mel: Artifact = serde_json::from_value(raw["log_mel"].clone()).unwrap();
            assert_eq!(mel.bytes, actual_mel * 128 * 4, "{}", case.id);
            if mel.path.is_some() {
                mel.read(&root, &root).unwrap();
            }
        }
        if let Some(encoder) = &case.encoder {
            assert_eq!(encoder.bytes, valid_encoder * 1024 * 4, "{}", case.id);
            if encoder.path.is_some() {
                encoder.read(&root, &root).unwrap();
            }
            if case.split.is_some() {
                corpus_tensors += 1;
            }
        }
    }
    assert_eq!(corpus_tensors, 32);
}

#[test]
fn speech_self_check_seal_pins_both_head_margins_and_budget() {
    let root = synapse_parity::parity_dir();
    let manifest = synapse_parity::manifest::Manifest::load(&root.join("models.json")).unwrap();
    let profile = manifest.profiles["parakeet-tdt-0.6b-v3.owned-metal"]
        .transcribe
        .as_ref()
        .unwrap();
    let raw = profile.self_check.read(&root, &root).unwrap();
    let subset: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let cases = subset["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    let mut total = raw.len();
    for case in cases {
        for key in [
            "minimum_token_margin",
            "minimum_duration_margin",
            "minimum_margin",
        ] {
            assert!(case[key].as_f64().unwrap() >= 2.0 * profile.tau_margin.unwrap());
        }
        let trace: Artifact = serde_json::from_value(case["trace"].clone()).unwrap();
        let trace = decode_trace(&trace.read(&root, &root).unwrap()).unwrap();
        let token_margin = trace
            .iter()
            .map(|s| f64::from(s.token_logit1) - f64::from(s.token_logit2))
            .fold(f64::INFINITY, f64::min);
        let duration_margin = trace
            .iter()
            .map(|s| f64::from(s.duration_logit1) - f64::from(s.duration_logit2))
            .fold(f64::INFINITY, f64::min);
        // Decimal JSON parsing may round a recorded f64 margin by one
        // representable floating-point step.
        // The actual trace margins must still meet the strict subset threshold.
        assert!((case["minimum_token_margin"].as_f64().unwrap() - token_margin).abs() <= 1e-12);
        assert!(
            (case["minimum_duration_margin"].as_f64().unwrap() - duration_margin).abs() <= 1e-12
        );
        assert!(token_margin >= 2.0 * profile.tau_margin.unwrap());
        assert!(duration_margin >= 2.0 * profile.tau_margin.unwrap());
        total += trace.len() * 40;
        if let Some(wav) = case.get("wav") {
            let wav: Artifact = serde_json::from_value(wav.clone()).unwrap();
            total += wav.read(&root, &root).unwrap().len();
        }
    }
    assert!(total <= 256 * 1024, "{total}");
    assert_eq!(subset["sealed_bytes"], total);
}

fn step(frame: i32, token: i32, duration: i32) -> TraceStep {
    TraceStep {
        frame_index: frame,
        token_top1: token,
        token_top2: if token == 1 { 2 } else { 1 },
        duration_top1: duration,
        duration_top2: if duration == 0 { 1 } else { 0 },
        token_logit1: 4.0,
        token_logit2: 0.0,
        duration_logit1: 4.0,
        duration_logit2: 0.0,
        emitted: token != BLANK,
        cap_hit: false,
    }
}

#[test]
fn speech_trace_rejects_shifted_frame_index() {
    let reference = vec![step(0, 1, 1), step(1, BLANK, 1)];
    let mut actual = reference.clone();
    actual[0].frame_index = 1;
    assert!(grade_trace(&reference, &actual, &[[1, 1, 1]], 2, 0.01).is_err());
}

#[test]
fn speech_trace_rejects_truncated_trace() {
    let reference = vec![step(0, 1, 1), step(1, BLANK, 1)];
    assert!(grade_trace(&reference, &reference[..1], &[[1, 0, 1]], 2, 0.01).is_err());
}

#[test]
fn speech_trace_rejects_extra_step() {
    let reference = vec![step(0, 1, 1), step(1, BLANK, 1)];
    let mut actual = reference.clone();
    actual.push(step(2, BLANK, 1));
    assert!(grade_trace(&reference, &actual, &[[1, 0, 1]], 2, 0.01).is_err());
}

#[test]
fn speech_trace_rejects_triples_inconsistent_with_trace() {
    let reference = vec![step(0, 1, 1), step(1, BLANK, 1)];
    assert!(grade_trace(&reference, &reference, &[[2, 0, 1]], 2, 0.01).is_err());
    assert!(!grade_trace(&reference, &reference, &[[1, 0, 1]], 2, 0.01).unwrap());
}

#[test]
fn speech_trace_exemption_is_strict_and_head_specific() {
    let mut reference = vec![step(0, 1, 1), step(1, BLANK, 1)];
    reference[0].token_logit1 = 0.005;
    let mut actual = reference.clone();
    actual[0].token_top1 = 3;
    assert!(grade_trace(&reference, &actual, &[[3, 0, 1]], 2, 0.01).unwrap());
    assert!(grade_trace(&reference, &actual, &[[3, 0, 1]], 2, 0.001).is_err());
    // A low token margin cannot excuse a differing duration head.
    actual[0].duration_top1 = 2;
    actual[0].duration_top2 = 0;
    actual.truncate(1);
    assert!(grade_trace(&reference, &actual, &[[3, 0, 2]], 2, 0.01).is_err());
    assert!(grade_trace(&reference, &actual, &[[1, 0, 2]], 2, 0.01).is_err());
}

#[test]
fn speech_trace_cap_only_forces_zero_advance() {
    let mut trace: Vec<_> = (0..10).map(|_| step(0, 1, 0)).collect();
    trace[9].cap_hit = true;
    assert_eq!(implied_triples(&trace, 1).unwrap(), vec![[1, 0, 0]; 10]);
    trace[9].cap_hit = false;
    trace[9].duration_top1 = 1;
    trace[9].duration_top2 = 0;
    assert_eq!(implied_triples(&trace, 1).unwrap().last(), Some(&[1, 0, 1]));
}

#[test]
fn speech_encoder_gate_checks_every_frame_and_finiteness() {
    let reference = vec![1.0; 2048];
    assert!((grade_encoder(&reference, &reference, 2).unwrap() - 1.0).abs() < 1e-12);
    let mut actual = reference.clone();
    actual[1024..].fill(-1.0);
    assert!(grade_encoder(&reference, &actual, 2).is_err());
    actual[0] = f32::NAN;
    assert!(grade_encoder(&reference, &actual, 2).is_err());
    assert!(grade_encoder(&reference, &reference[..1024], 2).is_err());
}

#[test]
fn speech_evaluator_requires_finite_encoder_observations_without_a_stored_tensor() {
    let root = synapse_parity::parity_dir();
    let manifest = synapse_parity::manifest::Manifest::load(&root.join("models.json")).unwrap();
    let profile = manifest.profiles["parakeet-tdt-0.6b-v3.owned-metal"]
        .transcribe
        .as_ref()
        .unwrap();
    let fixture = SpeechFixture {
        id: "missing-observation".into(),
        sample_count: 1280,
        duration_ms: 80,
        valid_encoder_frames: 1,
        trace: Artifact {
            path: Some("unread.trace".into()),
            cache_path: None,
            sha256: "f".repeat(64),
            bytes: 40,
        },
        encoder: None,
        text: "".into(),
        split: None,
        ground_truth: None,
    };
    let mut observations = std::collections::BTreeMap::from([(
        fixture.id.clone(),
        SpeechObservation {
            valid_encoder_frames: Some(1),
            ..Default::default()
        },
    )]);
    let error = evaluate_transcribe(
        profile,
        std::slice::from_ref(&fixture),
        &observations,
        &root,
        &root,
        "invalid_params",
    )
    .unwrap_err();
    assert!(error.to_string().contains("missing encoder observation"));
    for encoder in [vec![f32::NAN; 1024], vec![0.0; 512]] {
        observations.get_mut(&fixture.id).unwrap().encoder_output = Some(encoder);
        let error = evaluate_transcribe(
            profile,
            std::slice::from_ref(&fixture),
            &observations,
            &root,
            &root,
            "invalid_params",
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("malformed or nonfinite encoder observation"));
    }
}

#[test]
fn speech_wer_normalizer_and_per_split_gate() {
    assert_eq!(
        normalize_words("  Ｓtraße, L'AMI—世界！\nΣς "),
        ["strasse", "lami世界", "σσ"]
    );
    assert_eq!(
        word_edits(&normalize_words("a b c"), &normalize_words("a c d")),
        2
    );
    assert!(grade_wer([
        ("clean", "a b", "a b", "a b"),
        ("de", "eins zwei", "eins zwei", "eins")
    ])
    .is_err());
    assert_eq!(
        grade_wer([("clean", "a b", "a c", "a b")]).unwrap()["clean"],
        [0.5, 0.0]
    );
    // Word-weighted aggregation: one edit in 200 words is exactly the gate.
    let text = vec!["a"; 200].join(" ");
    let short = vec!["a"; 199].join(" ");
    assert!(grade_wer([("clean", text.as_str(), text.as_str(), short.as_str())]).is_ok());
}

#[test]
fn speech_profile_requires_all_three_tolerances() {
    let manifest =
        synapse_parity::manifest::Manifest::load(&synapse_parity::parity_dir().join("models.json"))
            .unwrap();
    let profile = manifest.profiles["parakeet-tdt-0.6b-v3.owned-metal"]
        .transcribe
        .as_ref()
        .unwrap();
    profile.validate().unwrap();
    for index in 0..3 {
        let mut missing = profile.clone();
        match index {
            0 => missing.tau_feat = None,
            1 => missing.tau_resample = None,
            _ => missing.tau_margin = None,
        }
        assert!(missing.validate().is_err());
    }
}

#[test]
fn speech_applicability_requires_exact_pre_engine_refusal() {
    let manifest =
        synapse_parity::manifest::Manifest::load(&synapse_parity::parity_dir().join("models.json"))
            .unwrap();
    let mut profile = manifest.profiles["parakeet-tdt-0.6b-v3.owned-metal"]
        .transcribe
        .clone()
        .unwrap();
    profile.max_duration_ms = 1000;
    let refusal = |details| SpeechRefusal {
        code: "invalid_params".into(),
        details,
    };
    assert!(grade_applicability(
        &profile,
        "invalid_params",
        1000,
        16000,
        Some(13),
        None,
        true
    )
    .unwrap());
    assert!(!grade_applicability(
        &profile,
        "invalid_params",
        1001,
        16016,
        None,
        Some(&refusal([1001, 1000, 13, 13])),
        false
    )
    .unwrap());
    assert!(grade_applicability(
        &profile,
        "invalid_params",
        1001,
        16016,
        Some(13),
        None,
        true
    )
    .is_err());
    assert!(grade_applicability(
        &profile,
        "invalid_params",
        1000,
        16000,
        None,
        Some(&refusal([1000, 1000, 13, 13])),
        false
    )
    .is_err());
    assert!(grade_applicability(
        &profile,
        "invalid_params",
        1000,
        16000,
        Some(12),
        None,
        true
    )
    .is_err());
    assert!(grade_applicability(
        &profile,
        "duration_limit",
        1001,
        16016,
        None,
        Some(&refusal([1001, 1000, 13, 13])),
        false
    )
    .is_err());
    // Sample-count admission is independent of the input's rounded duration.
    assert!(!grade_applicability(
        &profile,
        "invalid_params",
        1000,
        16800,
        None,
        Some(&refusal([1000, 1000, 14, 13])),
        false
    )
    .unwrap());
}
