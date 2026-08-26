#![forbid(unsafe_code)]

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use subc_client_rs::{CallError, CallOptions, ConsumerOptions, SubcConsumer};
use subc_protocol::{BindIdentity, RouteTarget};

fn default_subc_path() -> PathBuf {
    env::var_os("SYNAPSE_CONNECTION_FILE")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("XDG_RUNTIME_DIR")
                .map(PathBuf::from)
                .map(|directory| directory.join("synapse/subc-connection.json"))
        })
        .unwrap_or_else(|| env::temp_dir().join("synapse/subc-connection.json"))
}
const MODULE_ID: &str = "synapse";
const DEFAULT_COMPONENTS: usize = 8;

#[derive(Debug, Parser)]
#[command(
    name = "ck-synapse-opctl",
    about = "Drive Synapse operations through the fleet subc daemon"
)]
struct Cli {
    /// Path to the subc daemon connection file.
    #[arg(long, global = true, default_value_os_t = default_subc_path())]
    subc: PathBuf,

    /// Print unformatted response JSON instead of the operator view.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Probe {
        #[command(subcommand)]
        command: ProbeCommand,
    },
    Admission {
        #[command(subcommand)]
        command: AdmissionCommand,
    },
    Approvals {
        #[command(subcommand)]
        command: ApprovalsCommand,
    },
    Embed {
        #[command(subcommand)]
        command: EmbedCommand,
    },
    Job {
        #[command(subcommand)]
        command: JobCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelsCommand {
    /// List the catalog and current model states.
    List,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Submit a model manifest for loading.
    Load {
        /// JSON object or path to a file containing the model.load params object.
        #[arg(long)]
        manifest: String,
    },
    /// Show a model-load job or model runtime state.
    Status {
        /// A job_* id returned by model.load, or a model id.
        reference: String,
    },
}

#[derive(Debug, Subcommand)]
enum ProbeCommand {
    /// Start an explicit certification probe job.
    Run {
        /// Restrict the probe to one model/lane id.
        #[arg(long)]
        lane: Option<String>,
    },
    /// Show a probe job's state.
    Status {
        /// The job_* id returned by probe run.
        job_id: String,
    },
    /// Show certification, performance, and active assignment rows.
    Report,
}

#[derive(Debug, Subcommand)]
enum AdmissionCommand {
    /// Show the advisory scheduler snapshot.
    Status,
}

#[derive(Debug, Subcommand)]
enum ApprovalsCommand {
    /// Copy the checked-in retired owned-decode approval records into the
    /// approval table once; the migration is idempotent and preserves the
    /// records as the initial approval state.
    MigrateOwnedDecode,
    /// Create or explicitly re-enable one exact approval identity.
    Enable {
        #[arg(long)]
        model_id: String,
        #[arg(long)]
        decode_fingerprint: String,
        #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
        grammar_enabled: bool,
    },
    /// Disable one exact (model_id, decode_fingerprint) approval.
    Disable {
        #[arg(long)]
        model_id: String,
        #[arg(long)]
        decode_fingerprint: String,
        #[arg(long)]
        reason: String,
    },
    /// Disable every owned-decode approval atomically in a single transaction
    /// so a rollback cannot expose a partially updated approval set.
    EmergencyRollback {
        #[arg(long)]
        reason: String,
    },
}

#[derive(Debug, Subcommand)]
enum EmbedCommand {
    /// Submit an embedding batch.
    Batch(EmbedBatchArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "lower")]
enum InputType {
    Document,
    Query,
}

impl InputType {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Query => "query",
        }
    }
}

#[derive(Debug, Args)]
struct EmbedBatchArgs {
    /// Model id, alias, or fingerprint to require.
    #[arg(long)]
    model: String,

    /// Embedding configuration required by the consumer.
    #[arg(long, value_enum)]
    input_type: InputType,

    /// JSONL containing strings or objects with `id` and `text` fields.
    #[arg(long)]
    texts_file: Option<PathBuf>,

    /// Submit one text; repeat the option for multiple items.
    #[arg(long, action = clap::ArgAction::Append)]
    text: Vec<String>,

    /// Number of leading vector components shown in the operator view.
    #[arg(long, default_value_t = DEFAULT_COMPONENTS)]
    components: usize,
}

#[derive(Debug, Subcommand)]
enum JobCommand {
    /// Show the durable state (and page zero when it is already committed).
    Status { id: String },
    /// Read every result page currently committed for a job.
    Pages { id: String },
}

#[derive(Clone, Debug)]
struct SubmittedItem {
    id: String,
    text: String,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    if !cli.subc.is_file() {
        bail!(
            "subc connection file does not exist: {}",
            cli.subc.display()
        );
    }
    let project_root = discover_repo_root()?;
    let consumer = SubcConsumer::connect(&cli.subc, ConsumerOptions::default())
        .await
        .with_context(|| format!("connect to subc through {}", cli.subc.display()))?;
    let identity = BindIdentity {
        project_root,
        harness: "opctl".to_string(),
        session: format!("opctl-{}", std::process::id()),
    };

    let result = execute(&consumer, &identity, &cli.command).await;
    consumer.close().await;
    let output = result?;
    emit_output(&output, cli.json)?;
    ensure_no_error_envelope(&output)
}

async fn execute(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    command: &Command,
) -> Result<CommandOutput> {
    match command {
        Command::Models {
            command: ModelsCommand::List,
        } => call(consumer, identity, "models.list", json!({})).await,
        Command::Model { command } => match command {
            ModelCommand::Load { manifest } => {
                let params = parse_manifest_arg(manifest)?;
                call(consumer, identity, "model.load", params).await
            }
            ModelCommand::Status { reference } => {
                let params = if reference.starts_with("job_") {
                    json!({ "job_id": reference })
                } else {
                    json!({ "model_id": reference })
                };
                call(consumer, identity, "model.status", params).await
            }
        },
        Command::Probe { command } => match command {
            ProbeCommand::Run { lane } => {
                let params = lane
                    .as_ref()
                    .map(|lane| json!({ "models": [lane] }))
                    .unwrap_or_else(|| json!({}));
                call(consumer, identity, "probe.start", params).await
            }
            ProbeCommand::Status { job_id } => {
                call(
                    consumer,
                    identity,
                    "probe.status",
                    json!({ "job_id": job_id }),
                )
                .await
            }
            ProbeCommand::Report => call(consumer, identity, "probe.report", json!({})).await,
        },
        Command::Admission {
            command: AdmissionCommand::Status,
        } => call(consumer, identity, "admission.status", json!({})).await,
        Command::Approvals { command } => match command {
            ApprovalsCommand::MigrateOwnedDecode => {
                call(
                    consumer,
                    identity,
                    "approvals.migrate_owned_decode",
                    json!({
                        "seed_revision": "owned-decode-approval-migration-v1",
                        "schema_revision": "runtime-bound-records-contracts-v1",
                    }),
                )
                .await
            }
            ApprovalsCommand::Enable {
                model_id,
                decode_fingerprint,
                grammar_enabled,
            } => {
                call(
                    consumer,
                    identity,
                    "approvals.enable",
                    json!({
                        "model_id": model_id,
                        "decode_fingerprint": decode_fingerprint,
                        "grammar_enabled": grammar_enabled,
                    }),
                )
                .await
            }
            ApprovalsCommand::Disable {
                model_id,
                decode_fingerprint,
                reason,
            } => {
                call(
                    consumer,
                    identity,
                    "approvals.disable",
                    json!({
                        "model_id": model_id,
                        "decode_fingerprint": decode_fingerprint,
                        "reason": reason,
                    }),
                )
                .await
            }
            ApprovalsCommand::EmergencyRollback { reason } => {
                call(
                    consumer,
                    identity,
                    "approvals.emergency_rollback",
                    json!({ "reason": reason }),
                )
                .await
            }
        },
        Command::Embed {
            command: EmbedCommand::Batch(args),
        } => {
            let items = read_submitted_items(args)?;
            let wire_items = items
                .iter()
                .map(|item| json!({ "id": item.id, "text": item.text }))
                .collect::<Vec<_>>();
            let mut response = call(
                consumer,
                identity,
                "embed.batch",
                json!({
                    "model": args.model,
                    "input_type": args.input_type.as_str(),
                    "items": wire_items,
                }),
            )
            .await?;
            if response_error(response.primary()).is_none() {
                verify_embedding_response(response.primary(), &items)?;
            }
            response.embedding_components = Some(args.components);
            Ok(response)
        }
        Command::Job { command } => match command {
            JobCommand::Status { id } => {
                call(consumer, identity, "embed.result", json!({ "job_id": id })).await
            }
            JobCommand::Pages { id } => job_pages(consumer, identity, id).await,
        },
    }
}

#[derive(Debug)]
struct CommandOutput {
    responses: Vec<Value>,
    embedding_components: Option<usize>,
}

impl CommandOutput {
    fn one(response: Value) -> Self {
        Self {
            responses: vec![response],
            embedding_components: None,
        }
    }

    fn primary(&self) -> &Value {
        &self.responses[0]
    }

    fn json_value(&self) -> Value {
        if self.responses.len() == 1 {
            self.responses[0].clone()
        } else {
            Value::Array(self.responses.clone())
        }
    }
}

async fn call(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    method: &str,
    params: Value,
) -> Result<CommandOutput> {
    let request = serde_json::to_vec(&json!({ "method": method, "params": params }))
        .context("encode Synapse request")?;
    let timeout = if matches!(
        method,
        "model.load" | "model.status" | "probe.start" | "probe.status"
    ) {
        Duration::from_secs(180)
    } else {
        Duration::from_secs(60)
    };
    let options = CallOptions {
        timeout,
        ..CallOptions::default()
    };
    let bytes = consumer
        .call(
            RouteTarget::ManagementSurface {
                module_id: MODULE_ID.to_string(),
            },
            identity.clone(),
            request,
            options,
        )
        .await
        .map_err(format_call_error)?;
    let response = serde_json::from_slice(&bytes)
        .with_context(|| format!("decode {method} response as JSON"))?;
    Ok(CommandOutput::one(response))
}

async fn job_pages(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    job_id: &str,
) -> Result<CommandOutput> {
    let first = single_response(
        call(
            consumer,
            identity,
            "embed.result",
            json!({ "job_id": job_id, "page": 0 }),
        )
        .await?,
    )?;
    if response_error(&first).is_some() {
        return Ok(CommandOutput::one(first));
    }
    let available = first
        .pointer("/result/pages_available")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if available == 0 {
        return Ok(CommandOutput::one(first));
    }

    let mut pages = vec![first];
    for page in 1..available {
        let page = u32::try_from(page).context("job has more pages than the wire can address")?;
        let response = single_response(
            call(
                consumer,
                identity,
                "embed.result",
                json!({ "job_id": job_id, "page": page }),
            )
            .await?,
        )?;
        if response_error(&response).is_some() {
            pages.push(response);
            break;
        }
        pages.push(response);
    }
    Ok(CommandOutput {
        responses: pages,
        embedding_components: None,
    })
}

fn single_response(output: CommandOutput) -> Result<Value> {
    output
        .responses
        .into_iter()
        .next()
        .context("Synapse call returned no response envelope")
}

fn format_call_error(error: CallError) -> anyhow::Error {
    match error {
        CallError::Module(body) => {
            anyhow!("code={} layer=module message={}", body.code, body.message)
        }
        other => anyhow!("code=subc_call_failed layer=subc message={other}"),
    }
}

fn emit_output(output: &CommandOutput, raw_json: bool) -> Result<()> {
    if raw_json {
        println!("{}", serde_json::to_string(&output.json_value())?);
        return Ok(());
    }
    if let Some(components) = output.embedding_components {
        print_embedding_view(output.primary(), components);
    } else {
        println!("{}", serde_json::to_string_pretty(&output.json_value())?);
    }
    Ok(())
}

fn print_embedding_view(response: &Value, components: usize) {
    let result = response.get("result").unwrap_or(response);
    if response_error(response).is_some() || result.get("vectors").is_none() {
        println!(
            "{}",
            serde_json::to_string_pretty(response).unwrap_or_else(|_| response.to_string())
        );
        return;
    }

    println!(
        "fingerprint={} table_epoch={} dims={}",
        display_field(result.get("fingerprint")),
        display_field(result.get("table_epoch")),
        display_field(result.get("dims"))
    );
    if let Some(vectors) = result.get("vectors").and_then(Value::as_array) {
        for item in vectors {
            let shown = item
                .get("vector")
                .and_then(Value::as_array)
                .map(|vector| {
                    vector
                        .iter()
                        .take(components)
                        .map(Value::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            println!(
                "id={} content_sha256={} vector[0..{}]=[{}]",
                display_field(item.get("id")),
                display_field(item.get("content_sha256")),
                components.min(
                    item.get("vector")
                        .and_then(Value::as_array)
                        .map(Vec::len)
                        .unwrap_or(0)
                ),
                shown
            );
        }
    }
}

fn display_field(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
        None => "<missing>".to_string(),
    }
}

fn ensure_no_error_envelope(output: &CommandOutput) -> Result<()> {
    for response in &output.responses {
        if let Some(error) = response_error(response) {
            let code = error
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let layer = error
                .get("layer")
                .and_then(Value::as_str)
                .unwrap_or("application");
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("error envelope omitted a message");
            let class = error
                .get("class")
                .and_then(Value::as_str)
                .map(|class| format!(" class={class}"))
                .unwrap_or_default();
            bail!("code={code} layer={layer}{class} message={message}");
        }
    }
    Ok(())
}

fn response_error(response: &Value) -> Option<&Map<String, Value>> {
    response.pointer("/result/error").and_then(Value::as_object)
}

fn parse_manifest_arg(argument: &str) -> Result<Value> {
    let path = Path::new(argument);
    let value: Value = if path.is_file() {
        let bytes = fs::read(path)
            .with_context(|| format!("read model manifest from {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("parse model manifest from {}", path.display()))?
    } else {
        serde_json::from_str(argument).with_context(|| {
            format!("manifest is neither a readable file nor an inline JSON object: {argument}")
        })?
    };
    if !value.is_object() {
        bail!("model manifest must be a JSON object containing model.load params");
    }
    Ok(value)
}

fn read_submitted_items(args: &EmbedBatchArgs) -> Result<Vec<SubmittedItem>> {
    let mut items = Vec::new();
    let mut ids = HashSet::new();
    if let Some(path) = &args.texts_file {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("read embedding texts from {}", path.display()))?;
        for (line_index, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line).with_context(|| {
                format!(
                    "parse JSONL line {} from {}",
                    line_index + 1,
                    path.display()
                )
            })?;
            let generated_id = format!("item-{:04}", items.len());
            let item = match value {
                Value::String(text) => SubmittedItem {
                    id: generated_id,
                    text,
                },
                Value::Object(object) => {
                    let text = object
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            anyhow!(
                                "JSONL line {} in {} needs a string `text` field",
                                line_index + 1,
                                path.display()
                            )
                        })?
                        .to_string();
                    let id = object
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or(generated_id);
                    SubmittedItem { id, text }
                }
                _ => bail!(
                    "JSONL line {} in {} must be a string or an object",
                    line_index + 1,
                    path.display()
                ),
            };
            if !ids.insert(item.id.clone()) {
                bail!("duplicate embedding item id '{}'", item.id);
            }
            items.push(item);
        }
    }
    for text in &args.text {
        let id = format!("item-{:04}", items.len());
        ids.insert(id.clone());
        items.push(SubmittedItem {
            id,
            text: text.clone(),
        });
    }
    if items.is_empty() {
        bail!("embed batch requires --texts-file, at least one --text, or both");
    }
    Ok(items)
}

fn verify_embedding_response(response: &Value, submitted: &[SubmittedItem]) -> Result<()> {
    let Some(vectors) = response
        .pointer("/result/vectors")
        .and_then(Value::as_array)
    else {
        // A job-tier response has no vectors yet; `job pages` retrieves and displays them.
        return Ok(());
    };
    let fingerprint = response
        .pointer("/result/fingerprint")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .context("embedding response omitted a non-empty fingerprint")?;
    let _table_epoch = response
        .pointer("/result/table_epoch")
        .and_then(Value::as_u64)
        .context("embedding response omitted numeric table_epoch")?;
    let dims = response
        .pointer("/result/dims")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .context("embedding response omitted positive numeric dims")?;
    let by_id = submitted
        .iter()
        .map(|item| (item.id.as_str(), item.text.as_str()))
        .collect::<HashMap<_, _>>();
    if vectors.len() != submitted.len() {
        bail!(
            "embedding response item count {} does not match submitted count {}",
            vectors.len(),
            submitted.len()
        );
    }
    let mut seen = HashSet::with_capacity(vectors.len());
    for item in vectors {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .context("embedding response item omitted string id")?;
        if !seen.insert(id) {
            bail!("embedding response repeated item id '{id}'");
        }
        let text = by_id
            .get(id)
            .with_context(|| format!("embedding response returned unknown item id '{id}'"))?;
        let echoed = item
            .get("content_sha256")
            .and_then(Value::as_str)
            .with_context(|| format!("embedding response item '{id}' omitted content_sha256"))?;
        verify_content_echo(text, echoed)
            .with_context(|| format!("embedding response item '{id}' diverged"))?;

        let vector = item
            .get("vector")
            .and_then(Value::as_array)
            .with_context(|| format!("embedding response item '{id}' omitted vector"))?;
        if vector.len() != dims {
            bail!(
                "embedding response item '{id}' has {} components but envelope fingerprint {fingerprint} declares {dims}",
                vector.len()
            );
        }
        let mut nonzero = false;
        for (index, component) in vector.iter().enumerate() {
            let component = component.as_f64().with_context(|| {
                format!("embedding response item '{id}' component {index} is not numeric")
            })?;
            if !component.is_finite() {
                bail!("embedding response item '{id}' component {index} is not finite");
            }
            nonzero |= component != 0.0;
        }
        if !nonzero {
            bail!("embedding response item '{id}' vector is all zero");
        }
    }
    Ok(())
}

fn verify_content_echo(text: &str, echoed: &str) -> Result<String> {
    let expected = hex::encode(Sha256::digest(text.as_bytes()));
    if echoed != expected {
        bail!("content_sha256 mismatch: submitted={expected} echoed={echoed}");
    }
    Ok(expected)
}

fn discover_repo_root() -> Result<PathBuf> {
    let current = env::current_dir().context("read current directory")?;
    for candidate in current.ancestors() {
        if candidate.join(".git").exists() && candidate.join("Cargo.toml").is_file() {
            return candidate
                .canonicalize()
                .with_context(|| format!("canonicalize repository root {}", candidate.display()));
        }
    }
    bail!(
        "current directory {} is not inside a Git Cargo workspace; run opctl from the Synapse repository",
        current.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_echo_accepts_exact_submitted_utf8() {
        let text = "memory: café ☕\nsecond line";
        let expected = hex::encode(Sha256::digest(text.as_bytes()));
        assert_eq!(verify_content_echo(text, &expected).unwrap(), expected);
    }

    #[test]
    fn content_echo_rejects_changed_content() {
        let echoed = hex::encode(Sha256::digest(b"normalized text"));
        let error = verify_content_echo("original text", &echoed).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("content_sha256 mismatch"));
        assert!(message.contains("submitted="));
        assert!(message.contains("echoed="));
    }

    #[test]
    fn manifest_argument_parses_inline_object() {
        let value = parse_manifest_arg(r#"{"source":"file","engine":"owned-metal"}"#).unwrap();
        assert_eq!(value["source"], "file");
        assert_eq!(value["engine"], "owned-metal");
    }

    #[test]
    fn manifest_argument_reads_json_file() {
        let path = env::temp_dir().join(format!(
            "synapse-opctl-manifest-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::write(
            &path,
            r#"{"source":"file","files":{"model":"model.safetensors"}}"#,
        )
        .unwrap();
        let value = parse_manifest_arg(path.to_str().unwrap()).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(value["source"], "file");
        assert_eq!(value["files"]["model"], "model.safetensors");
    }

    #[test]
    fn manifest_argument_rejects_non_object_json() {
        let error = parse_manifest_arg("[]").unwrap_err();
        assert!(error.to_string().contains("must be a JSON object"));
    }

    #[test]
    fn clap_keeps_inline_manifest_as_one_argument() {
        let cli = Cli::try_parse_from([
            "ck-synapse-opctl",
            "model",
            "load",
            "--manifest",
            r#"{"source":"file"}"#,
        ])
        .unwrap();
        let Command::Model {
            command: ModelCommand::Load { manifest },
        } = cli.command
        else {
            panic!("expected model load command");
        };
        assert_eq!(manifest, r#"{"source":"file"}"#);
    }
}
