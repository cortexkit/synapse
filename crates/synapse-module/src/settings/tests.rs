use super::*;
use subc_client_rs::config::{ConfigLayer, ConfigLayerSnapshot, ConfigStatus};

const LIVE_CONFIG: &str = r#"{"microllm_max_tokens": 512, "preload_models": []}"#;

fn request(contents: Option<&str>, base: Option<ConfigView>) -> ConfigReviewRequest {
    ConfigReviewRequest::new(
        base.as_ref().map_or(0, |view| view.generation),
        vec![ConfigLayerSnapshot::file(
            DEFAULT_MODULE_ID,
            ConfigLayer::User,
            "/config/cortexkit/synapse.jsonc",
            contents.map(|s| s.as_bytes().to_vec()),
        )],
        base,
    )
}

fn effective(request: &ConfigReviewRequest) -> ConfigView {
    let layers = review_layers(request).unwrap();
    let tuple = schema()
        .review_tuple(&layers, &ValidationContext::new(DEFAULT_MODULE_ID, None))
        .unwrap();
    ConfigView::new(
        0,
        tuple.effective,
        tuple.sources,
        vec![],
        ConfigStatus::Unstamped,
    )
}

#[test]
fn settings_schema_is_valid_and_matches_manifest() {
    let declared = declaration(DEFAULT_MODULE_ID);
    let checked = Schema::new(declared.schema).expect("valid daemon settings schema");
    let manifest = manifest(DEFAULT_MODULE_ID);
    let schema = manifest.provides.iter().find_map(|role| match role {
        ProviderRole::ManagementSurface { config_schema, .. } => Some(config_schema),
        _ => None,
    });
    assert_eq!(schema, Some(checked.normalized()));
    assert!(declared.project_dirs.is_empty());
    assert!(Path::new(&declared.user_file).ends_with(Path::new("cortexkit").join("synapse.jsonc")));
}

#[test]
fn settings_review_accepts_live_file_and_decodes_to_legacy_config() {
    // The live file sets exactly the default values. The daemon still needs
    // rows for both keys because their sources change from default to user.
    let base = effective(&request(None, None));
    let request = request(Some(LIVE_CONFIG), Some(base));
    let view = effective(&request);
    let reply = review(DEFAULT_MODULE_ID, request);
    assert!(reply.refusals.is_empty(), "{:?}", reply.refusals);
    assert_eq!(
        reply
            .fields
            .iter()
            .map(|row| row.path.as_str())
            .collect::<Vec<_>>(),
        ["/microllm_max_tokens", "/preload_models"]
    );
    assert!(reply
        .fields
        .iter()
        .all(|row| row.old_source == Layer::Default
            && row.new_source == Layer::User
            && row.apply == ApplyKind::ModuleRestart));
    assert_eq!(
        decode_config(view.effective, "managed settings").unwrap(),
        parse_module_config_json(LIVE_CONFIG, "legacy settings").unwrap(),
    );
}

#[test]
fn settings_without_user_file_review_to_startup_defaults() {
    for missing in [Some("{}"), None] {
        let request = request(missing, None);
        let view = effective(&request);
        let reply = review(DEFAULT_MODULE_ID, request);
        assert!(reply.refusals.is_empty(), "{:?}", reply.refusals);
        assert_eq!(reply.fields.len(), view.sources.len());
        assert!(reply
            .fields
            .iter()
            .all(|row| row.new_source == Layer::Default));
        assert_eq!(
            decode_config(view.effective, "managed settings").unwrap(),
            ModuleConfig::default()
        );
    }
}

#[test]
fn settings_review_names_each_retired_key() {
    let retired = [
        ("alias_admin_enabled", "alias_admin_enabled was removed (found 'alias_admin_enabled'); aliases are no longer supported"),
        ("grammar_enabled", "grammar_enabled was removed; owned decode serving was retired"),
        ("sidecar_spec", "sidecar_spec was removed; owned decode serving was retired"),
        ("decode_chain_k", "decode_chain_k was removed; owned decode serving was retired"),
        ("remote_providers", "remote_providers were removed; modules should call providers directly"),
    ];
    for (key, reason) in retired {
        let input = format!(r#"{{"{key}":false}}"#);
        let reply = review(DEFAULT_MODULE_ID, request(Some(&input), None));
        assert!(reply.fields.is_empty());
        assert_eq!(reply.refusals.len(), 1);
        assert_eq!(reply.refusals[0].paths, [format!("/{key}")]);
        assert_eq!(reply.refusals[0].reasons, [reason]);
    }
}

#[test]
fn settings_review_names_bad_type() {
    let request = request(Some(r#"{"worker":{"load_timeout_ms":"fast"}}"#), None);
    let reply = review(DEFAULT_MODULE_ID, request);
    assert!(reply.fields.is_empty());
    assert_eq!(reply.refusals.len(), 1);
    assert_eq!(reply.refusals[0].code, "value_invalid");
    assert_eq!(reply.refusals[0].paths, ["/worker/load_timeout_ms"]);
    assert!(reply.refusals[0].reasons[0].contains("load_timeout_ms"));
}

#[test]
fn settings_preload_defaults_are_not_inserted_and_task_aliases_are_refused() {
    let mut user = json!({"preload_models":[{
        "engine":"llama", "model_id":"llama-generate",
        "model_path":"model", "tokenizer_path":"tokenizer"
    }]});
    let good = request(Some(&user.to_string()), None);
    let view = effective(&good);
    assert_eq!(view.effective["preload_models"], user["preload_models"]);
    let reply = review(DEFAULT_MODULE_ID, good);
    assert!(reply.refusals.is_empty(), "{:?}", reply.refusals);

    for alias in ["kind", "capability"] {
        user["preload_models"][0][alias] = json!("generate");
        let refuse = review(DEFAULT_MODULE_ID, request(Some(&user.to_string()), None));
        assert_eq!(
            refuse.refusals[0].paths,
            [format!("/preload_models/0/{alias}")]
        );
        user["preload_models"][0]
            .as_object_mut()
            .unwrap()
            .remove(alias);
    }
}
