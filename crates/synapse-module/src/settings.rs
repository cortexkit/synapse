//! Synapse startup settings and daemon configuration hooks.

#[cfg(test)]
mod tests;

use super::*;
use subc_client_rs::config::{
    ApplyKind, ConfigDeclareRequest, ConfigReviewRefusal, ConfigReviewReply, ConfigReviewRequest,
    ConfigState, ConfigView, FieldChange, Layer, LayerValue, RecordedValue, Schema,
    ValidationContext, CONFIG_GATE_CAPABILITY,
};
use tokio::sync::watch;

pub(super) const SYNAPSE_CONFIG_PATH_ENV: &str = "SYNAPSE_CONFIG_PATH";

// Every setting carries a `title`: the desktop's settings page uses it as the
// field's label, and falls back to the raw key when it is missing.
fn field(kind: &str, title: &str, description: &str, default: impl Into<Value>) -> Value {
    json!({"type": kind, "title": title, "description": description, "default": default.into(),
        "x-ck-apply": "module_restart", "x-ck-layers": ["user"]})
}

fn unsigned(title: &str, description: &str, default: impl Into<Value>) -> Value {
    let mut field = field("integer", title, description, default);
    field["minimum"] = json!(0);
    field
}

fn u32_field(title: &str, description: &str, default: u32) -> Value {
    let mut field = unsigned(title, description, default);
    field["maximum"] = json!(u32::MAX);
    field
}

fn section(title: &str, description: &str, properties: Value) -> Value {
    let defaults = properties
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, schema)| (name.clone(), schema["default"].clone()))
        .collect::<serde_json::Map<_, _>>();
    let mut schema = field("object", title, description, defaults);
    schema["unevaluatedProperties"] = json!(false);
    schema["properties"] = properties;
    schema
}

fn preload_schema() -> Value {
    // Defaults on item properties are annotations only: the gate replaces the
    // whole array without merging item defaults. Omitted options stay omitted.
    let text = |title, description| field("string", title, description, "");
    let required_text = |title, description| {
        let mut schema = text(title, description);
        schema["x-ck-required"] = json!(true);
        schema
    };
    let mut schema = field(
        "array",
        "Startup models",
        "Models to register during startup.",
        json!([]),
    );
    schema["items"] = json!({
        "type": "object", "title": "Startup model", "unevaluatedProperties": false,
        "properties": {
            "engine": required_text("Engine", "Engine that serves the model."),
            "model_path": required_text("Model path", "Path to the model artifact."),
            "tokenizer_path": required_text("Tokenizer path", "Path to the tokenizer artifact."),
            "model_id": text("Model ID", "Model identifier; if omitted, derived from the engine and list position."),
            "profile": text("Profile", "Manifest profile; required for embedding and reranking preloads."),
            "prompt_template": text("Prompt template", "Prompt template for the model."),
            "task": text("Task", "Model task; if omitted, inferred from the engine."),
            "artifact_digest": text("Artifact digest", "Expected digest of the model artifact."),
            "format": text("Format", "Model artifact format."),
            "pooling": text("Pooling", "Embedding pooling strategy."),
            "normalize": field("boolean", "Normalize vectors", "Normalize embedding vectors.", false),
            "max_tokens": unsigned("Context limit", "Model context limit in tokens.", 0),
            "quant": text("Quantization", "Model quantization label."),
            "backend": text("Backend", "Backend name for the model."),
            "family": text("Architecture family", "Model architecture family."),
            "dtype": text("Data type", "Numerical data type."),
            "kernel_revision": text("Kernel revision", "Kernel revision for the model."),
            "ptx_virtual_arch": text("CUDA PTX architecture", "CUDA PTX virtual architecture."),
            "minimum_device_cc": field("number", "Minimum CUDA compute capability", "Minimum CUDA compute capability.", 0),
            "minimum_cuda_driver_api": u32_field("Minimum CUDA driver", "Minimum CUDA driver API version.", 0),
            "execution": text("Execution mode", "Execution mode for the model."),
            "attention_units": unsigned("Attention units", "Attention unit count for the model.", 0),
            "worker_bin": text("Worker executable", "Path to the worker executable."),
            "worker_runtime_dir": text("Worker runtime folder", "Path to the worker runtime directory.")
        }
    });
    schema
}

pub(super) fn schema() -> &'static Schema {
    static SCHEMA: OnceLock<Schema> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        let mut knob = field("string", "Performance policy", "Machine-wide performance policy.", PerfKnob::default().as_str());
        knob["enum"] = json!(["performance", "balanced", "quiet"]);
        let raw = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Synapse",
            "type": "object", "unevaluatedProperties": false, "x-ck-layers": ["user"],
            "properties": {
                "hf_endpoint": field("string", "Model download address", "Base URL for Hugging Face model downloads.", default_hf_endpoint()),
                "cache_max_bytes": unsigned("Model cache size", "Model cache size budget in bytes.", DEFAULT_CACHE_MAX_BYTES),
                "microllm_max_tokens": u32_field("Generation token limit", "Maximum tokens for MicroLLM generation.", DEFAULT_MICROLLM_MAX_TOKENS),
                "knob": knob,
                "inline": section("Direct requests", "Limits for synchronous model requests.", json!({
                    "max_items": unsigned("Items per request", "Maximum items in one inline request.", DEFAULT_INLINE_MAX_ITEMS),
                    "max_tokens": unsigned("Tokens per request", "Maximum input tokens in one inline request.", DEFAULT_INLINE_MAX_TOKENS),
                    "byte_budget": unsigned("In-flight byte budget", "Byte budget for in-flight inline requests.", DEFAULT_INLINE_BYTE_BUDGET),
                    "max_queue_ms": unsigned("Queue wait limit (ms)", "Maximum time to queue an inline request, in milliseconds.", DEFAULT_MAX_QUEUE_MS),
                    "deadline_ms": unsigned("Request deadline (ms)", "Inline request deadline in milliseconds.", DEFAULT_DEADLINE_MS),
                    "estimated_execution_ms": unsigned("Estimated run time (ms)", "Estimated inline execution time for admission, in milliseconds.", DEFAULT_ESTIMATED_EXECUTION_MS),
                    "max_concurrent_workers": unsigned("Concurrent workers", "Maximum concurrent inline workers.", DEFAULT_MAX_CONCURRENT_WORKERS)
                })),
                "jobs": section("Background jobs", "Limits and retention for background jobs.", json!({
                    "execution_ttl_ms": unsigned("Run time limit (ms)", "Time allowed for job execution, in milliseconds.", DEFAULT_JOB_EXECUTION_TTL_MS),
                    "result_retention_ttl_ms": unsigned("Result retention (ms)", "Time to retain completed job results, in milliseconds.", DEFAULT_JOB_RESULT_RETENTION_TTL_MS),
                    "result_page_bytes": unsigned("Result page size", "Target size of a job result page, in bytes.", DEFAULT_JOB_RESULT_PAGE_BYTES),
                    "bulk_quantum_tokens": unsigned("Bulk scheduling quantum", "Token quantum when scheduling bulk jobs.", DEFAULT_JOB_BULK_QUANTUM_TOKENS)
                })),
                "worker": section("Workers", "Worker process settings.", json!({
                    "load_timeout_ms": unsigned("Model load timeout (ms)", "Worker model load timeout in milliseconds.", DEFAULT_WORKER_LOAD_TIMEOUT_MS)
                })),
                "log": section("Logging", "Performance and worker logging settings.", json!({
                    "perf_interval_secs": unsigned("Performance summary interval (s)", "Interval for performance summaries in seconds; zero disables them.", 0),
                    "worker_forward_lines_per_sec": u32_field("Worker log lines per second", "Maximum forwarded log lines per second from a worker.", DEFAULT_WORKER_FORWARD_LINES_PER_SEC)
                })),
                "ane_direct": section("Apple Neural Engine", "Direct Apple Neural Engine serving settings.", json!({
                    "tight_packing": field("boolean", "Tight packing", "Enable tight packing on the direct ANE lane.", false)
                })),
                "certify_observation": field("boolean", "Certification observations", "Disclose engine-bound observations for the release certification runner only.", false),
                "preload_models": preload_schema()
            }
        });
        Schema::new(raw).expect("Synapse settings schema must be valid")
    })
}

fn user_config_path(module_id: &str) -> PathBuf {
    PathBuf::from(cortexkit_store_types::resolve_config_home())
        .join("cortexkit")
        .join(format!("{module_id}.jsonc"))
}

pub(super) fn declaration(module_id: &str) -> ConfigDeclareRequest {
    ConfigDeclareRequest::new(
        schema().normalized().clone(),
        user_config_path(module_id).to_string_lossy(),
        vec![],
    )
}

pub(super) fn load_legacy_config() -> Result<ModuleConfig, ModuleError> {
    // Isolated certification and test daemons do not advertise config.gate.
    if let Some(path) = env::var_os(SYNAPSE_CONFIG_PATH_ENV) {
        return load_config_file(Path::new(&path));
    }
    let path = user_config_path(DEFAULT_MODULE_ID);
    if !path.is_absolute() {
        return Err(ModuleError::Config(format!(
            "config path {} is relative; set XDG_CONFIG_HOME to an absolute path (or provide an absolute HOME / Windows APPDATA or USERPROFILE)",
            path.display()
        )));
    }
    match fs::read_to_string(&path) {
        Ok(contents) => parse_module_config_json(&contents, &path.to_string_lossy()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(ModuleConfig::default()),
        Err(error) => Err(ModuleError::Config(format!(
            "read {}: {error}",
            path.display()
        ))),
    }
}

fn load_config_file(path: &Path) -> Result<ModuleConfig, ModuleError> {
    let contents = fs::read_to_string(path)
        .map_err(|error| ModuleError::Config(format!("read {}: {error}", path.display())))?;
    parse_module_config_json(&contents, &path.to_string_lossy())
}

pub(super) fn parse_module_config_json(
    contents: &str,
    source: &str,
) -> Result<ModuleConfig, ModuleError> {
    let value = serde_json::from_str(&strip_json_comments(contents))?;
    decode_config(value, source)
}

fn retired_setting(value: &Value) -> Option<(String, String)> {
    if value.get("remote_providers").is_some() {
        return Some((
            "/remote_providers".into(),
            "remote_providers were removed; modules should call providers directly".into(),
        ));
    }
    for (prefix, object) in [("", value), ("/dev", &value["dev"])] {
        for key in [
            "alias_admin_enabled",
            "dev_alias_admin",
            "enable_alias_admin",
        ] {
            if object.get(key).is_some() {
                return Some((format!("{prefix}/{key}"), format!(
                    "alias_admin_enabled was removed (found '{key}'); aliases are no longer supported"
                )));
            }
        }
    }
    for key in ["grammar_enabled", "sidecar_spec", "decode_chain_k"] {
        if value.get(key).is_some() {
            return Some((
                format!("/{key}"),
                format!("{key} was removed; owned decode serving was retired"),
            ));
        }
    }
    if let Some(preloads) = value.get("preload_models").and_then(Value::as_array) {
        for (index, preload) in preloads.iter().enumerate() {
            let Some(engine) = preload.get("engine").and_then(Value::as_str) else {
                continue;
            };
            if canonical_engine_name(engine) == "owned-metal-decode" {
                let model_id = preload
                    .get("model_id")
                    .and_then(Value::as_str)
                    .map_or_else(|| format!("{engine}-{index}"), str::to_owned);
                return Some((
                    format!("/preload_models/{index}/engine"),
                    format!("preload model_id '{model_id}': owned decode serving was retired"),
                ));
            }
        }
    }
    None
}

fn decode_config(value: Value, source: &str) -> Result<ModuleConfig, ModuleError> {
    if let Some((_, reason)) = retired_setting(&value) {
        return Err(ModuleError::Config(reason));
    }
    let config: ModuleConfig = serde_json::from_value(value).map_err(|error| {
        if let Some(field) = unknown_field_from_json_error(&error) {
            ModuleError::Config(format!(
                "unknown config field '{field}' in {source} (deny_unknown_fields)"
            ))
        } else {
            ModuleError::Json(error)
        }
    })?;
    for (index, preload) in config.preload_models.iter().enumerate() {
        let engine = canonical_engine_name(&preload.engine);
        let model_id = preload
            .model_id
            .clone()
            .unwrap_or_else(|| format!("{}-{index}", preload.engine));
        if preload.profile.is_none()
            && matches!(
                parse_model_task(preload.task.as_deref(), &engine, &model_id)?,
                ModelTask::Embed | ModelTask::Rerank
            )
        {
            return Err(ModuleError::Config(format!(
                "preload model_id '{model_id}': free-form preloads for embed and rerank were removed; specify a profile"
            )));
        }
    }
    Ok(config)
}

pub(super) fn review(module_id: &str, request: ConfigReviewRequest) -> ConfigReviewReply {
    let layers = review_layers(&request).and_then(|layers| {
        schema()
            .review_tuple(&layers, &ValidationContext::new(module_id, None))
            .map_err(|refusal| {
                let code = serde_json::to_value(refusal.code).expect("refusal code serializes");
                let reason = format!("invalid Synapse configuration at {}", refusal.path);
                refuse(
                    code.as_str().expect("refusal code is a string"),
                    refusal.path,
                    reason,
                )
            })
    });
    let fields = layers.and_then(|tuple| {
        let config =
            decode_config(tuple.effective.clone(), "managed settings").map_err(|error| {
                let path = if matches!(error, ModuleError::Config(_)) {
                    "/preload_models"
                } else {
                    ""
                };
                refuse("value_invalid", path, error.to_string())
            })?;
        validate_hf_endpoint(&config.hf_endpoint).map_err(|_| {
            refuse(
                "value_invalid",
                "/hf_endpoint",
                "invalid Hugging Face endpoint URL",
            )
        })?;
        Ok(field_changes(
            module_id,
            request.base.as_ref(),
            &tuple.effective,
            &tuple.sources,
        ))
    });
    match fields {
        Ok(fields) => ConfigReviewReply::new(fields, vec![]),
        Err(refusal) => ConfigReviewReply::new(vec![], vec![refusal]),
    }
}

fn refuse(code: &str, path: impl Into<String>, reason: impl Into<String>) -> ConfigReviewRefusal {
    ConfigReviewRefusal::new(code, vec![path.into()], vec![reason.into()])
}

fn review_layers(request: &ConfigReviewRequest) -> Result<Vec<LayerValue>, ConfigReviewRefusal> {
    request
        .layers
        .iter()
        .filter_map(|layer| layer.bytes.as_deref().map(|bytes| (layer, bytes)))
        .map(|(snapshot, bytes)| {
            let layer = snapshot
                .layer
                .known_layer()
                .ok_or_else(|| refuse("layer_forbidden", "", "unknown configuration layer"))?;
            // The daemon normalizes the user JSONC before supplying the snapshot.
            let value = serde_json::from_slice(bytes)
                .map_err(|_| refuse("value_invalid", "", "invalid Synapse settings JSON"))?;
            if let Some((path, reason)) = retired_setting(&value) {
                return Err(refuse("value_invalid", path, reason));
            }
            Ok(LayerValue::new(layer, value))
        })
        .collect()
}

fn field_changes(
    module_id: &str,
    base: Option<&ConfigView>,
    effective: &Value,
    sources: &BTreeMap<String, Layer>,
) -> Vec<FieldChange> {
    let old_effective = base.map(|view| &view.effective);
    let source = |sources: &BTreeMap<String, Layer>, path: &str| {
        sources.get(path).copied().unwrap_or(Layer::Default)
    };
    leaf_paths(old_effective, effective)
        .into_iter()
        .filter_map(|path| {
            let old = old_effective.and_then(|value| value.pointer(&path));
            let new = effective.pointer(&path);
            let old_source = base.map_or(Layer::Default, |view| source(&view.sources, &path));
            let new_source = source(sources, &path);
            if old == new && old_source == new_source {
                return None;
            }
            let record =
                |value: Option<&Value>| RecordedValue::new(value.cloned().unwrap_or(Value::Null));
            let mut field = FieldChange::new(
                module_id,
                &path,
                record(old),
                record(new),
                ApplyKind::ModuleRestart,
            );
            field.old_source = old_source;
            field.new_source = new_source;
            Some(field)
        })
        .collect()
}

fn leaf_paths(old: Option<&Value>, new: &Value) -> BTreeSet<String> {
    // Match the gate's changeset boundary: descend through objects, but a
    // preload array is one setting and is replaced as a unit.
    fn visit(value: &Value, path: &str, paths: &mut BTreeSet<String>) {
        if let Some(object) = value.as_object().filter(|object| !object.is_empty()) {
            for (key, value) in object {
                visit(
                    value,
                    &format!("{path}/{}", key.replace('~', "~0").replace('/', "~1")),
                    paths,
                );
            }
        } else if !path.is_empty() {
            paths.insert(path.to_owned());
        }
    }
    let mut paths = BTreeSet::new();
    if let Some(old) = old {
        visit(old, "", &mut paths);
    }
    visit(new, "", &mut paths);
    paths
}

type BootResult = Result<(), String>;

pub(super) struct Startup {
    pub(super) ack: OnceLock<ModuleHelloAckBody>,
    effective: OnceLock<Option<Value>>,
    completion: watch::Sender<Option<BootResult>>,
    last_status: Mutex<Option<&'static str>>,
}

impl Default for Startup {
    fn default() -> Self {
        Self {
            ack: OnceLock::new(),
            effective: OnceLock::new(),
            completion: watch::Sender::new(None),
            last_status: Mutex::new(None),
        }
    }
}

impl Startup {
    fn managed(&self) -> bool {
        self.ack.get().is_some_and(|ack| {
            ack.subc_capabilities
                .iter()
                .any(|capability| capability == CONFIG_GATE_CAPABILITY)
        })
    }

    pub(super) fn observe(&self, state: ConfigState) {
        let status = state.as_str();
        let mut last = self.last_status.lock().expect("config state");
        if last.replace(status) != Some(status)
            && matches!(
                state,
                ConfigState::Failed { .. }
                    | ConfigState::ConfigGated { .. }
                    | ConfigState::NotDeliveryTarget { .. }
                    | ConfigState::AnchorPending { .. }
            )
        {
            tracing::warn!(target: "config", status, "daemon settings are not ready");
        }
    }

    pub(super) async fn completed(&self) -> BootResult {
        self.completion
            .subscribe()
            .wait_for(Option::is_some)
            .await
            .expect("boot completion sender is held by Startup")
            .as_ref()
            .expect("completed boot")
            .clone()
    }

    fn finish(&self, result: BootResult) {
        if let Err(error) = &result {
            tracing::error!(target: "config", %error, "synapse boot failed after HELLO_ACK");
        }
        self.completion.send_replace(Some(result));
    }
}

impl SynapseHandler {
    async fn start_once(
        &self,
        effective: Option<Value>,
        config: impl FnOnce() -> Result<ModuleConfig, ModuleError> + Send + 'static,
    ) -> Result<(), Vec<String>> {
        let startup = &self.inner.startup;
        let chosen = startup.effective.get_or_init(|| {
            let handler = self.clone();
            // Own boot independently of a waiting callback so cancellation cannot
            // start it twice or leave a partially initialized runtime in use.
            tokio::spawn(async move {
                let worker = handler.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let ack = worker
                        .inner
                        .startup
                        .ack
                        .get()
                        .expect("boot follows HELLO_ACK");
                    let config = config().map_err(|error| error.to_string())?;
                    let state = worker
                        .initialize(ack, config)
                        .map_err(|error| error.to_string())?;
                    let _ = worker.inner.state.set(Arc::clone(&state));
                    diagnostics_findings::start(&state);
                    start_catalog_revalidation(Arc::clone(&state));
                    media_jobs::start_maintenance(&state);
                    let _ = start_perf_sampler(state);
                    Ok(())
                })
                .await
                .unwrap_or_else(|error| Err(error.to_string()));
                handler.inner.startup.finish(result);
            });
            effective.clone()
        });
        if let (Some(chosen), Some(effective)) = (chosen, &effective) {
            // The SDK currently can replay the old active view before delivering
            // the pending view on a restart child. Until it delivers pending
            // first, refuse a change rather than acknowledge unapplied settings.
            let changes = restart_required(chosen, effective);
            if !changes.is_empty() {
                return Err(changes);
            }
        }
        startup
            .completed()
            .await
            .map_err(|_| vec!["Synapse boot failed; see module log".into()])
    }

    pub(super) async fn apply_settings(&self, view: ConfigView) -> Result<(), Vec<String>> {
        if view.project_id.is_some() {
            return Err(vec!["Synapse has no project settings".into()]);
        }
        let config = decode_config(view.effective.clone(), "managed settings")
            .map_err(|_| vec!["invalid Synapse settings in delivered effective view".into()])?;
        validate_hf_endpoint(&config.hf_endpoint)
            .map_err(|_| vec!["/hf_endpoint: invalid Hugging Face endpoint URL".into()])?;
        self.start_once(Some(view.effective), move || Ok(config))
            .await
    }

    pub(super) async fn apply_legacy_settings(&self) {
        let _ = self.start_once(None, load_legacy_config).await;
    }

    pub(super) async fn serve_with_settings(
        &self,
        handle: subc_client_rs::ModuleHandle,
        mut serving: subc_client_rs::ModuleServeFuture,
        connection_file: PathBuf,
    ) -> Result<(), ModuleError> {
        tokio::select! {
            result = &mut serving => return result.map_err(ModuleError::Serve),
            boot = self.inner.startup.completed() => match boot {
                Ok(()) => event_outbox::start(&self.state().expect("boot installed state"), handle, connection_file),
                // Managed boot failure must leave the connection up to send the NACK.
                Err(error) if !self.inner.startup.managed() => return Err(ModuleError::Config(error)),
                Err(_) => {},
            }
        }
        serving.await.map_err(ModuleError::Serve)
    }
}

fn restart_required(old: &Value, new: &Value) -> Vec<String> {
    leaf_paths(Some(old), new)
        .into_iter()
        .filter(|path| old.pointer(path) != new.pointer(path))
        .map(|path| format!("{path}: setting requires a Synapse restart"))
        .collect()
}

fn unknown_field_from_json_error(error: &serde_json::Error) -> Option<String> {
    let message = error.to_string();
    let marker = "unknown field `";
    let start = message.find(marker)? + marker.len();
    let rest = &message[start..];
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
}

fn strip_json_comments(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(ch) = chars.next() {
        if in_string {
            output.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            output.push(ch);
            continue;
        }
        if ch == '/' {
            match chars.peek().copied() {
                Some('/') => {
                    let _ = chars.next();
                    for next in chars.by_ref() {
                        if next == '\n' {
                            output.push('\n');
                            break;
                        }
                    }
                    continue;
                }
                Some('*') => {
                    let _ = chars.next();
                    let mut previous = '\0';
                    for next in chars.by_ref() {
                        if previous == '*' && next == '/' {
                            break;
                        }
                        previous = next;
                    }
                    continue;
                }
                _ => {}
            }
        }
        output.push(ch);
    }
    output
}
