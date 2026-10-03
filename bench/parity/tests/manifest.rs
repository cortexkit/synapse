//! Tests over the committed `bench/parity/models.json`.

use std::collections::BTreeMap;

use serde_json::{json, Value};
use synapse_parity::arch::{check_against_config, check_schema};
use synapse_parity::canonical::sha256_hex;
use synapse_parity::hadamard;
use synapse_parity::manifest::{
    profile_id, AllMarker, DType, Fp32Tensors, Lane, Manifest, ToleranceClass, MANIFEST_FILE,
    MODEL_SLUGS,
};
use synapse_parity::oracle::{check_template, parse_oracle};
use synapse_parity::parity_dir;
use synapse_parity::rules::derived_profile;
use synapse_parity::safetensors::parse_header_json;
use synapse_parity::validate::{check_head, validate_dir, validate_manifest};
use synapse_parity::vulkan::{vulkan_floors, FLOOR_SEQUENCES};

fn manifest() -> Manifest {
    Manifest::load(&parity_dir().join(MANIFEST_FILE)).unwrap()
}

fn read_json(relative: &str) -> Value {
    serde_json::from_slice(&std::fs::read(parity_dir().join(relative)).unwrap()).unwrap()
}

fn tensor_index(slug: &str) -> BTreeMap<String, Vec<u64>> {
    let bytes =
        std::fs::read(parity_dir().join(format!("checkpoints/{slug}/tensor-index.json"))).unwrap();
    parse_header_json(&bytes)
        .unwrap()
        .tensors
        .into_iter()
        .map(|(name, info)| (name, info.shape))
        .collect()
}

fn expect_err<T: std::fmt::Debug>(result: synapse_parity::Result<T>, needle: &str) {
    let error = result.expect_err("expected a failure");
    assert!(
        error.0.contains(needle),
        "error {:?} does not mention {needle:?}",
        error.0
    );
}

#[test]
fn committed_manifest_passes_schema_only_validation() {
    validate_dir(&parity_dir()).unwrap();
}

#[test]
fn admission_and_reference_constants_are_pinned() {
    let m = manifest();
    assert_eq!(m.admission.max_context_tokens, 8192);
    assert_eq!(m.admission.ane_resident_shapes_per_model, 4);
    assert_eq!(m.admission.ane_resident_shapes_total, 8);
    assert_eq!(m.reference.reference_transformers_version, "5.16.1");
    assert_eq!(m.reference.reference_seed, 0);

    let mut changed = m.clone();
    changed.admission.ane_resident_shapes_total = 9;
    expect_err(
        validate_manifest(&changed, &parity_dir()),
        "admission constants",
    );
}

#[test]
fn every_model_pins_revision_tokenizer_operation_output_and_parameters() {
    let m = manifest();
    for slug in MODEL_SLUGS {
        let model = m.model(slug).unwrap();
        assert_eq!(model.hf_revision.len(), 40, "{slug}");
        assert!(model.tokenizer_digest.starts_with("sha256:"), "{slug}");
        assert!(model.output.dimension > 0, "{slug}");
        check_schema(slug, &model.architecture).unwrap();
    }
    assert_eq!(
        m.model("gte-modernbert-base").unwrap().hf_revision,
        "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
    );
}

#[test]
fn a_missing_config_value_fails_the_generator() {
    let m = manifest();
    let model = m.model("qwen3-embedding-0.6b").unwrap();
    let tokenizer_config = read_json("checkpoints/qwen3-embedding-0.6b/tokenizer_config.json");
    let mut config = read_json("checkpoints/qwen3-embedding-0.6b/config.json");
    check_against_config("q", &model.architecture, &config, Some(&tokenizer_config)).unwrap();
    config
        .as_object_mut()
        .unwrap()
        .remove("num_key_value_heads");
    expect_err(
        check_against_config("q", &model.architecture, &config, Some(&tokenizer_config)),
        "has no `num_key_value_heads`",
    );
    // The Qwen3 pad id comes from tokenizer_config.json; without it the
    // generator fails rather than assuming one.
    let config = read_json("checkpoints/qwen3-embedding-0.6b/config.json");
    expect_err(
        check_against_config("q", &model.architecture, &config, None),
        "tokenizer_config.json",
    );
}

#[test]
fn a_disagreeing_config_value_fails_the_generator() {
    let m = manifest();
    for (slug, key, value) in [
        ("qwen3-reranker-0.6b", "head_dim", json!(64)),
        ("qwen3-reranker-0.6b", "rope_theta", json!(10000)),
        ("gte-modernbert-base", "local_attention", json!(256)),
        (
            "gte-modernbert-base",
            "global_attn_every_n_layers",
            json!(2),
        ),
        (
            "gte-reranker-modernbert-base",
            "local_rope_theta",
            json!(20000.0),
        ),
        ("gte-reranker-modernbert-base", "norm_eps", json!(1e-6)),
    ] {
        let model = m.model(slug).unwrap();
        let tokenizer_config = model
            .files
            .contains_key("tokenizer_config.json")
            .then(|| read_json(&format!("checkpoints/{slug}/tokenizer_config.json")));
        let mut config = read_json(&format!("checkpoints/{slug}/config.json"));
        config[key] = value;
        expect_err(
            check_against_config(
                slug,
                &model.architecture,
                &config,
                tokenizer_config.as_ref(),
            ),
            "disagrees with the pinned checkpoint value",
        );
    }
}

#[test]
fn an_unknown_architecture_fails_and_the_causal_mask_comes_from_the_table() {
    let m = manifest();
    let model = m.model("qwen3-reranker-0.6b").unwrap();
    let tokenizer_config = read_json("checkpoints/qwen3-reranker-0.6b/tokenizer_config.json");
    let mut config = read_json("checkpoints/qwen3-reranker-0.6b/config.json");
    config["architectures"] = json!(["Qwen3ForSequenceClassification"]);
    expect_err(
        check_against_config("q", &model.architecture, &config, Some(&tokenizer_config)),
        "unknown architecture",
    );

    assert_eq!(model.architecture.params["attention_mask"], json!("causal"));
    let mut bidirectional = model.architecture.clone();
    bidirectional
        .params
        .insert("attention_mask".into(), json!("bidirectional"));
    expect_err(check_schema("q", &bidirectional), "architectures table");

    let mut extra = model.architecture.clone();
    extra.params.insert("sliding_window".into(), json!(4096));
    expect_err(check_schema("q", &extra), "not in the");
}

#[test]
fn template_literals_equal_the_oracle_and_any_difference_fails() {
    let m = manifest();
    let template = m
        .model("qwen3-reranker-0.6b")
        .unwrap()
        .grammar
        .template
        .clone()
        .unwrap();
    let readme = std::fs::read_to_string(parity_dir().join(&template.oracle.path)).unwrap();
    check_template(&template, &readme).unwrap();
    let oracle = parse_oracle(&readme).unwrap();
    assert!(oracle
        .prefix
        .starts_with("<|im_start|>system\nJudge whether the Document"));
    assert_eq!(
        oracle.suffix,
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );

    for field in 0..4 {
        let mut mutated = template.clone();
        let target = [
            &mut mutated.prefix,
            &mut mutated.instruction,
            &mut mutated.body_format,
            &mut mutated.suffix,
        ]
        .into_iter()
        .nth(field)
        .unwrap();
        target.push(' ');
        expect_err(check_template(&mutated, &readme), "qwen_template_mismatch");
    }
    let edited = readme.replacen("Note that the answer", "Note that answers", 1);
    expect_err(check_template(&template, &edited), "qwen_template_mismatch");
}

#[test]
fn yes_and_no_ids_are_recorded_and_checked() {
    let m = manifest();
    let readout = &m.model("qwen3-reranker-0.6b").unwrap().grammar.readout;
    assert_eq!(readout.yes.as_ref().unwrap().id, 9693);
    assert_eq!(readout.no.as_ref().unwrap().id, 2152);

    let mut swapped = m.clone();
    let model = swapped.models.get_mut("qwen3-reranker-0.6b").unwrap();
    let yes = model.grammar.readout.yes.clone();
    model.grammar.readout.yes = model.grammar.readout.no.clone();
    model.grammar.readout.no = yes;
    expect_err(
        validate_manifest(&swapped, &parity_dir()),
        "qwen_readout_mismatch",
    );
}

#[test]
fn rerank_heads_are_checked_against_the_tensor_index() {
    let m = manifest();
    let gte = m.model("gte-reranker-modernbert-base").unwrap();
    let gte_config = read_json("checkpoints/gte-reranker-modernbert-base/config.json");
    let gte_index = tensor_index("gte-reranker-modernbert-base");
    check_head(gte, &gte_config, &gte_index).unwrap();
    let head = gte.head.as_ref().unwrap();
    assert_eq!(
        head.required_tensor_keys,
        [
            "classifier.bias",
            "classifier.weight",
            "head.dense.weight",
            "head.norm.weight"
        ]
    );

    let mut missing = gte_index.clone();
    missing.remove("classifier.bias");
    expect_err(
        check_head(gte, &gte_config, &missing),
        "head_tensor_missing",
    );
    let mut reshaped = gte_index.clone();
    reshaped.insert("head.dense.weight".into(), vec![768, 1024]);
    expect_err(check_head(gte, &gte_config, &reshaped), "head tensor");
    let mut pooled = gte_config.clone();
    pooled["classifier_pooling"] = json!("cls");
    expect_err(
        check_head(gte, &pooled, &gte_index),
        "rerank head disagrees",
    );

    let qwen = m.model("qwen3-reranker-0.6b").unwrap();
    let qwen_config = read_json("checkpoints/qwen3-reranker-0.6b/config.json");
    let qwen_index = tensor_index("qwen3-reranker-0.6b");
    check_head(qwen, &qwen_config, &qwen_index).unwrap();
    assert_eq!(
        qwen.grammar.readout.weight.as_deref(),
        Some("model.embed_tokens.weight")
    );
    // Untied, the readout must be lm_head.weight.
    let mut untied = qwen.clone();
    untied
        .architecture
        .params
        .insert("tie_word_embeddings".into(), json!(false));
    expect_err(
        check_head(&untied, &qwen_config, &qwen_index),
        "readout weight must be `lm_head.weight`",
    );
    untied.grammar.readout.weight = Some("lm_head.weight".into());
    expect_err(
        check_head(&untied, &qwen_config, &qwen_index),
        "rerank head disagrees",
    );
}

#[test]
fn tolerance_class_and_fp32_lists_follow_the_rule() {
    let m = manifest();
    let mut fp32_class = Vec::new();
    for (id, profile) in &m.profiles {
        let model = m.model(&profile.model).unwrap();
        let derived = derived_profile(&profile.model, model, profile.lane).unwrap();
        assert_eq!(profile.storage_dtype, derived.storage_dtype, "{id}");
        assert_eq!(profile.compute_dtype, derived.compute_dtype, "{id}");
        assert_eq!(profile.fp32_tensors, derived.fp32_tensors, "{id}");
        assert_eq!(
            profile.rerank_tolerance_class, derived.rerank_tolerance_class,
            "{id}"
        );
        if profile.rerank_tolerance_class == ToleranceClass::Fp32 {
            fp32_class.push(id.clone());
        }
        if matches!(profile.lane, Lane::OwnedCuda | Lane::OwnedVulkan) {
            assert_eq!(profile.fp32_tensors, Fp32Tensors::List(vec![]), "{id}");
            assert_eq!(profile.storage_dtype, DType::F16, "{id}");
        }
    }
    // The expected values below are written out by hand from the rule in
    // src/rules.rs (only the f32 Metal gte-reranker profile is fp32-class;
    // direct ANE keeps exactly its CPU-stage tensors in fp32), so a bug in that
    // rule cannot also make this test pass.
    assert_eq!(fp32_class, ["gte-reranker-modernbert-base.owned-metal"]);
    assert_eq!(
        m.profiles["gte-reranker-modernbert-base.owned-metal"].fp32_tensors,
        Fp32Tensors::All(AllMarker::All)
    );
    assert_eq!(
        m.profiles["qwen3-reranker-0.6b.ane-direct-worker"].fp32_tensors,
        Fp32Tensors::List(vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into()
        ])
    );
    assert_eq!(
        m.profiles["gte-reranker-modernbert-base.ane-direct-worker"].fp32_tensors,
        Fp32Tensors::List(
            [
                "classifier.bias",
                "classifier.weight",
                "head.dense.weight",
                "head.norm.weight",
                "model.embeddings.tok_embeddings.weight",
                "rotation_in.weight",
                "rotation_out.weight",
            ]
            .map(String::from)
            .to_vec()
        )
    );

    let mut changed = m.clone();
    changed
        .profiles
        .get_mut("gte-reranker-modernbert-base.owned-metal")
        .unwrap()
        .rerank_tolerance_class = ToleranceClass::Fp16;
    expect_err(
        validate_manifest(&changed, &parity_dir()),
        "numeric-profile rule",
    );
    let mut changed = m.clone();
    changed
        .profiles
        .get_mut("qwen3-embedding-0.6b.ane-direct-worker")
        .unwrap()
        .fp32_tensors = Fp32Tensors::List(vec!["embed_tokens.weight".into()]);
    expect_err(
        validate_manifest(&changed, &parity_dir()),
        "numeric-profile rule",
    );
}

#[test]
fn direct_ane_profiles_pin_rotation_and_gelu_lowering() {
    let m = manifest();
    for slug in MODEL_SLUGS {
        let profile = &m.profiles[&profile_id(slug, Lane::AneDirect)];
        if slug.starts_with("qwen3") {
            assert_eq!(profile.rotation.as_deref(), Some("none"), "{slug}");
        } else {
            assert_eq!(
                profile.rotation.as_deref(),
                Some("modernbert-hadamard-768-v2"),
                "{slug}"
            );
            assert_eq!(profile.gelu_lowering.as_deref(), Some("tanh"), "{slug}");
        }
    }
    let rotation = &m.rotations["modernbert-hadamard-768-v2"];
    // Pinned here as a literal: an independent Python rebuild of the
    // documented construction produced this digest, so a change to the
    // generator cannot also move the expectation.
    assert_eq!(
        rotation.sha256,
        "5cd34f0d01ce33615f987bd62be88b8e595f5083a3c94b42291c053d5cf87d53"
    );
    assert_eq!(hadamard::digest(rotation.sign_seed), rotation.sha256);
}

#[test]
fn worker_profiles_carry_a_package_digest_and_metal_profiles_do_not() {
    let m = manifest();
    for (id, profile) in &m.profiles {
        assert_eq!(
            profile.converted_package_digest.is_some(),
            profile.lane.is_worker(),
            "{id}"
        );
    }
    let mut changed = m.clone();
    changed
        .profiles
        .get_mut("gte-modernbert-base.owned-metal")
        .unwrap()
        .converted_package_digest = Some(format!("sha256:{}", "1".repeat(64)));
    expect_err(validate_manifest(&changed, &parity_dir()), "must not carry");
}

#[test]
fn vulkan_floors_equal_the_sizing_function() {
    let m = manifest();
    for slug in MODEL_SLUGS {
        let id = profile_id(slug, Lane::OwnedVulkan);
        let profile = &m.profiles[&id];
        assert_eq!(profile.vulkan_sub_batch_max_tokens, Some(8192), "{id}");
        let floors = vulkan_floors(
            m.model(slug).unwrap(),
            profile.storage_dtype,
            &profile.fp32_tensors,
            8192,
            256,
            8192,
        )
        .unwrap();
        assert_eq!(
            profile.vulkan_min_storage_buffer_range,
            Some(floors.min_storage_buffer_range),
            "{id}"
        );
        assert_eq!(
            profile.vulkan_min_device_local_bytes,
            Some(floors.min_device_local_bytes),
            "{id}"
        );
        // The device-local floor depends on the sequence count (the result
        // buffer holds every sequence), so sizing for 255 instead of 256 must
        // give a smaller value.
        let fewer = vulkan_floors(
            m.model(slug).unwrap(),
            profile.storage_dtype,
            &profile.fp32_tensors,
            8192,
            255,
            8192,
        )
        .unwrap();
        assert!(
            fewer.min_device_local_bytes < floors.min_device_local_bytes,
            "{id}"
        );
    }
    assert_eq!(FLOOR_SEQUENCES, 256);
    // The largest Qwen3 buffer is its f16 token-embedding table.
    assert_eq!(
        m.profiles["qwen3-embedding-0.6b.owned-vulkan"].vulkan_min_storage_buffer_range,
        Some(151669 * 1024 * 2)
    );

    let mut changed = m.clone();
    let profile = changed
        .profiles
        .get_mut("qwen3-reranker-0.6b.owned-vulkan")
        .unwrap();
    profile.vulkan_min_device_local_bytes = profile.vulkan_min_device_local_bytes.map(|v| v - 1);
    expect_err(
        validate_manifest(&changed, &parity_dir()),
        "sizing function",
    );
    assert!(vulkan_floors(
        m.model("gte-modernbert-base").unwrap(),
        DType::F16,
        &Fp32Tensors::List(vec![]),
        8192,
        256,
        4096
    )
    .is_err());
}

#[test]
fn digests_cover_every_grammar_and_profile_entry() {
    let m = manifest();
    assert_eq!(m.digests.grammar.len(), 4);
    assert_eq!(m.digests.profiles.len(), 16);
    let mut all: Vec<&String> = m
        .digests
        .grammar
        .values()
        .chain(m.digests.profiles.values())
        .collect();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 20, "every entry hashes differently");
    assert_eq!(m.computed_digests().unwrap(), m.digests);

    // Changing a per-profile key changes that profile's digest only.
    let mut gelu = m.clone();
    gelu.profiles
        .get_mut("gte-modernbert-base.ane-direct-worker")
        .unwrap()
        .gelu_lowering = Some("erf".into());
    let digests = gelu.computed_digests().unwrap();
    for (id, digest) in &digests.profiles {
        assert_eq!(
            digest != &m.digests.profiles[id],
            id == "gte-modernbert-base.ane-direct-worker",
            "{id}"
        );
    }

    // Template literals, yes/no ids and architecture parameters change the
    // grammar digest (where they are grammar) and every profile of the model.
    let edits: [(&str, fn(&mut Manifest)); 3] = [
        ("template", |m| {
            m.models
                .get_mut("qwen3-reranker-0.6b")
                .unwrap()
                .grammar
                .template
                .as_mut()
                .unwrap()
                .instruction
                .push('!')
        }),
        ("yes id", |m| {
            m.models
                .get_mut("qwen3-reranker-0.6b")
                .unwrap()
                .grammar
                .readout
                .yes
                .as_mut()
                .unwrap()
                .id += 1
        }),
        ("architecture", |m| {
            m.models
                .get_mut("qwen3-reranker-0.6b")
                .unwrap()
                .architecture
                .params
                .insert("rope_theta".into(), json!(2000000.0));
        }),
    ];
    for (name, edit) in edits {
        let mut edited = m.clone();
        edit(&mut edited);
        let digests = edited.computed_digests().unwrap();
        for lane in Lane::ALL {
            let id = profile_id("qwen3-reranker-0.6b", lane);
            assert_ne!(
                digests.profiles[&id], m.digests.profiles[&id],
                "{name} {id}"
            );
        }
        assert_eq!(
            digests.profiles["qwen3-embedding-0.6b.owned-cuda"],
            m.digests.profiles["qwen3-embedding-0.6b.owned-cuda"]
        );
        if name != "architecture" {
            assert_ne!(
                digests.grammar["qwen3-reranker-0.6b"], m.digests.grammar["qwen3-reranker-0.6b"],
                "{name}"
            );
        }
    }

    // `rotation: none` and an absent rotation hash differently.
    let mut absent = m.clone();
    absent
        .profiles
        .get_mut("qwen3-embedding-0.6b.ane-direct-worker")
        .unwrap()
        .rotation = None;
    assert_ne!(
        absent
            .profile_digest("qwen3-embedding-0.6b.ane-direct-worker")
            .unwrap(),
        m.digests.profiles["qwen3-embedding-0.6b.ane-direct-worker"]
    );

    let mut stale = m.clone();
    stale
        .digests
        .grammar
        .insert("gte-modernbert-base".into(), sha256_hex(b"stale"));
    expect_err(
        validate_manifest(&stale, &parity_dir()),
        "recorded digests differ",
    );
}

#[test]
fn schema_validation_rejects_unknown_keys_and_noncanonical_bytes() {
    let mut value: Value =
        serde_json::from_slice(&std::fs::read(parity_dir().join(MANIFEST_FILE)).unwrap()).unwrap();
    value["profiles"]["gte-modernbert-base.owned-cuda"]["kernel"] = json!("x");
    expect_err(
        Manifest::from_slice(&serde_json::to_vec(&value).unwrap()),
        "unknown field",
    );

    let dir = std::env::temp_dir().join(format!("parity-noncanonical-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = std::fs::read(parity_dir().join(MANIFEST_FILE)).unwrap();
    let compact = serde_json::to_vec(&serde_json::from_slice::<Value>(&bytes).unwrap()).unwrap();
    std::fs::write(dir.join(MANIFEST_FILE), compact).unwrap();
    for entry in ["checkpoints", "oracles"] {
        copy_dir(&parity_dir().join(entry), &dir.join(entry));
    }
    expect_err(validate_dir(&dir), "not in canonical form");
    std::fs::remove_dir_all(&dir).unwrap();
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Full checks against the downloaded pinned checkpoints (digests of every
/// pinned file, the real tensor index, and the pinned tokenizer's ids for
/// every grammar token, with `yes` and `no` each a single id). Needs the
/// checkpoints in a Hugging Face hub cache named by `PARITY_HF_CACHE`, so it
/// is opt-in: `PARITY_HF_CACHE=~/.cache/huggingface/hub cargo test -- --ignored`.
#[test]
#[ignore = "needs the pinned checkpoints; set PARITY_HF_CACHE and run with --ignored"]
fn pinned_checkpoints_match_the_manifest() {
    let cache =
        std::env::var("PARITY_HF_CACHE").expect("PARITY_HF_CACHE names a Hugging Face hub cache");
    let m = manifest();
    for slug in MODEL_SLUGS {
        let dir = synapse_parity::checkpoint::snapshot_dir(
            std::path::Path::new(&cache),
            m.model(slug).unwrap(),
        );
        synapse_parity::checkpoint::check_checkpoint(&m, slug, &dir, &parity_dir()).unwrap();
    }
}
