//! Tests over the row fixtures, machine registry, release asset inventory,
//! preload regression inputs and reference fixture index.

use serde_json::{json, Value};
use synapse_parity::canonical::sha256_file;
use synapse_parity::inventory::{
    check_machine_registry, check_release_assets, check_row_fixture, load_json, module_defaults,
    Binding, MachineRegistry, ReleaseAssets, ROW_IDS,
};
use synapse_parity::manifest::{Manifest, MANIFEST_FILE, MODEL_SLUGS};
use synapse_parity::preload::{check_preload, PreloadInputs, PRELOAD_MODEL_IDS};
use synapse_parity::validate::validate_inventory;
use synapse_parity::{parity_dir, repo_root};

fn defaults() -> std::collections::BTreeMap<String, std::collections::BTreeMap<String, u64>> {
    module_defaults(
        &std::fs::read_to_string(repo_root().join("crates/synapse-module/src/lib.rs")).unwrap(),
    )
    .unwrap()
}

fn registry() -> MachineRegistry {
    load_json(&parity_dir().join("machines.json")).unwrap()
}

fn assets() -> ReleaseAssets {
    load_json(&parity_dir().join("release-assets.json")).unwrap()
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
fn committed_inventory_validates() {
    validate_inventory(&parity_dir()).unwrap();
}

#[test]
fn module_defaults_are_read_from_the_module_source() {
    let defaults = defaults();
    // Spot values written out by hand from crates/synapse-module/src/lib.rs.
    assert_eq!(defaults["inline"]["max_items"], 64);
    assert_eq!(defaults["inline"]["max_tokens"], 8192);
    assert_eq!(defaults["inline"]["byte_budget"], 64 * 1024 * 1024);
    assert_eq!(defaults["jobs"]["bulk_quantum_tokens"], 3072);
    assert_eq!(defaults["jobs"]["execution_ttl_ms"], 24 * 60 * 60 * 1000);
    assert_eq!(defaults["inline"].len(), 7);
    assert_eq!(defaults["jobs"].len(), 5);
}

#[test]
fn every_row_has_a_default_config_fixture_and_a_differing_value_fails() {
    let defaults = defaults();
    for row in ROW_IDS {
        let fixture: Value = load_json(&parity_dir().join(format!("rows/{row}.json"))).unwrap();
        check_row_fixture(row, &fixture, &defaults).unwrap();
        for (section, key) in [("inline", "max_items"), ("jobs", "bulk_quantum_tokens")] {
            let mut changed = fixture.clone();
            changed[section][key] = json!(changed[section][key].as_u64().unwrap() + 1);
            expect_err(
                check_row_fixture(row, &changed, &defaults),
                "differ from the module defaults",
            );
            let mut missing = fixture.clone();
            missing[section].as_object_mut().unwrap().remove(key);
            expect_err(
                check_row_fixture(row, &missing, &defaults),
                "differ from the module defaults",
            );
        }
    }
}

#[test]
fn the_registry_names_each_rows_machine() {
    let registry = registry();
    check_machine_registry(&registry, &parity_dir()).unwrap();
    assert_eq!(registry.rows["metal-m5"].machine, "m5-max-macbook-pro");
    assert_eq!(registry.rows["vulkan-linux-amd"].machine, "rog-ally-x");
    assert_eq!(registry.rows["vulkan-windows-amd"].machine, "rog-ally-x");
    assert_eq!(
        registry.rows["cuda-linux-nvidia"].machine,
        "vast-ai-linux-nvidia"
    );
    assert_eq!(
        registry.machines["m5-max-macbook-pro"]
            .model_identifier
            .as_deref(),
        Some("Mac17,6")
    );

    let mut unknown = registry.clone();
    unknown.rows.get_mut("ane-m5").unwrap().machine = "nowhere".into();
    expect_err(
        check_machine_registry(&unknown, &parity_dir()),
        "unknown machine",
    );
    let mut probe = registry.clone();
    probe.rows.get_mut("metal-m5").unwrap().floor_probe = true;
    expect_err(check_machine_registry(&probe, &parity_dir()), "floor_probe");
}

#[test]
fn release_assets_give_every_candidate_exactly_one_entry() {
    let registry = registry();
    let inventory = assets();
    check_release_assets(&inventory, &registry).unwrap();

    let find = |inventory: &ReleaseAssets, asset: &str| {
        inventory
            .assets
            .iter()
            .position(|a| a.asset == asset)
            .unwrap()
    };
    let bound = &inventory.assets[find(&inventory, "ck-synapse-linux-x64.zip")];
    assert_eq!(
        bound.rows,
        [
            "cuda-linux-nvidia",
            "vulkan-linux-amd",
            "vulkan-linux-nvidia"
        ]
    );
    let windows_cuda =
        &inventory.assets[find(&inventory, "ck-synapse-worker-cuda-windows-x64.zip")];
    assert_eq!(
        windows_cuda.binary.as_deref(),
        Some("ck-synapse-worker-cuda.exe")
    );
    assert_eq!(
        windows_cuda.runtime_files_from.as_deref(),
        Some("manifest.json")
    );

    // An undeclared new worker fails.
    let mut undeclared = inventory.clone();
    undeclared
        .assets
        .remove(find(&undeclared, "ck-synapse-worker-vulkan-linux-x64.zip"));
    expect_err(check_release_assets(&undeclared, &registry), "has no entry");

    // Marking the Vulkan worker exempt fails.
    let mut exempt = inventory.clone();
    let index = find(&exempt, "ck-synapse-worker-vulkan-windows-x64.zip");
    exempt.assets[index].binding = Binding::Exempt;
    exempt.assets[index].rows.clear();
    exempt.assets[index].reason = Some("not needed".into());
    expect_err(check_release_assets(&exempt, &registry), "cannot be exempt");

    // A duplicate entry, a missing bound row and an asset nobody builds fail.
    let mut duplicate = inventory.clone();
    duplicate.assets.push(duplicate.assets[0].clone());
    expect_err(
        check_release_assets(&duplicate, &registry),
        "more than one entry",
    );
    let mut rows = inventory.clone();
    let index = find(&rows, "ck-synapse-darwin-arm64.zip");
    rows.assets[index].rows.retain(|row| row != "ane-m5");
    expect_err(
        check_release_assets(&rows, &registry),
        "rows must be exactly",
    );
    let mut stray = inventory.clone();
    let mut extra = stray.assets[0].clone();
    extra.asset = "ck-synapse-worker-metal-darwin-arm64.zip".into();
    stray.assets.push(extra);
    expect_err(
        check_release_assets(&stray, &registry),
        "names no candidate asset",
    );
    let mut reasonless = inventory.clone();
    let index = find(&reasonless, "release-manifest.json");
    reasonless.assets[index].reason = None;
    expect_err(
        check_release_assets(&reasonless, &registry),
        "gives a reason",
    );
}

#[test]
fn preload_inputs_reproduce_the_production_fingerprints() {
    for model_id in PRELOAD_MODEL_IDS {
        let inputs: PreloadInputs =
            load_json(&parity_dir().join(format!("preload/{model_id}.json"))).unwrap();
        check_preload(&inputs).unwrap();
        let expected = inputs.expected_fingerprint.clone().unwrap();
        match model_id {
            "gte-reranker-modernbert-base-f32" => {
                assert_eq!(
                    expected,
                    "2fa5f24c0208f30c6db4bf18eb66bc0b2c46f765882cb744fcc58cd080b2b92d"
                )
            }
            _ => assert!(expected.starts_with("24cc5271f42d") && expected.len() == 64),
        }

        // Fails, never skips, when the expected fingerprint is absent.
        let mut absent = inputs.clone();
        absent.expected_fingerprint = None;
        expect_err(check_preload(&absent), "expected_fingerprint is absent");
        // Any changed input changes the fingerprint.
        let mut changed = inputs.clone();
        changed.max_tokens -= 1;
        assert!(check_preload(&changed).is_err());
        let mut changed = inputs.clone();
        changed.inline.insert("max_items".into(), 63);
        assert!(check_preload(&changed).is_err());
    }
}

#[test]
fn reference_fixtures_cover_every_model_with_digests() {
    let manifest = Manifest::load(&parity_dir().join(MANIFEST_FILE)).unwrap();
    let index: Value = load_json(&parity_dir().join("fixtures/index.json")).unwrap();
    let index = index.as_object().unwrap();
    assert_eq!(index.len(), MODEL_SLUGS.len());
    for slug in MODEL_SLUGS {
        let id = format!(
            "{slug}.ref-v1.transformers-{}.seed-{}",
            manifest.reference.reference_transformers_version, manifest.reference.reference_seed
        );
        let entry = &index[&id];
        let path = parity_dir().join(entry["path"].as_str().unwrap());
        assert_eq!(
            sha256_file(&path).unwrap(),
            entry["sha256"].as_str().unwrap(),
            "{id}"
        );
        let document: Value = load_json(&path).unwrap();
        assert_eq!(document["fixture_set_id"], json!(id));
        assert_eq!(
            document["reference"]["transformers_version"],
            json!("5.16.1")
        );
        assert_eq!(document["reference"]["seed"], json!(0));
        let cases = document["cases"].as_array().unwrap();
        let has = |category: &str| cases.iter().any(|case| case["category"] == json!(category));
        for category in ["short", "shape_boundary", "batched", "long"] {
            assert!(has(category), "{id} lacks {category}");
        }
        if manifest.model(slug).unwrap().operation == synapse_parity::manifest::Operation::Rerank {
            let pool = |name: &str| {
                cases
                    .iter()
                    .filter(|case| case["pool"] == json!(name))
                    .count()
            };
            assert_eq!((pool("pool-10"), pool("pool-100")), (10, 100), "{id}");
        }
        let long: Vec<usize> = cases
            .iter()
            .filter(|case| case["category"] == json!("long"))
            .map(|case| case["input_ids"].as_array().unwrap().len())
            .collect();
        assert_eq!(long, [8192], "{id}");
    }
}
