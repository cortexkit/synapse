#![forbid(unsafe_code)]

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
    process::ExitCode,
    time::Duration,
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use subc_client_rs::{discover, CallError, CallOptions, ConsumerOptions, SubcConsumer};
use subc_protocol::{BindIdentity, RouteTarget};

mod model_download;

const MODULE_ID: &str = "synapse";
const DEFAULT_COMPONENTS: usize = 8;

#[derive(Debug, Parser)]
#[command(
    name = "ck-synapse-opctl",
    version,
    about = "Drive Synapse operations through the fleet subc daemon"
)]
struct Cli {
    /// Explicit subc connection file. Otherwise use SUBC_CONNECTION_FILE or daemon discovery.
    #[arg(long, global = true)]
    subc: Option<PathBuf>,

    /// Print unformatted response JSON instead of the operator view.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Temporarily lend Neural Engine lanes to a private measurement daemon.
    Lanes {
        #[command(subcommand)]
        command: LanesCommand,
    },
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    Admission {
        #[command(subcommand)]
        command: AdmissionCommand,
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
    /// Download a catalog model and watch until it is installed.
    Download { catalog_id: String },
    /// Remove an installed catalog model.
    Remove { catalog_id: String },
}

#[derive(Debug, Subcommand)]
enum LanesCommand {
    /// Fence ANE admission, drain requests, unload workers, and release the lock.
    Pause {
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=120))]
        minutes: u64,
        #[arg(long)]
        reason: String,
    },
    /// End the pause early; admission opens only after the lock is re-acquired.
    Resume,
    /// Wait for hardware_free=true before starting a private daemon.
    Status,
}

fn lane_request(command: &LanesCommand) -> (&'static str, Value) {
    match command {
        LanesCommand::Pause { minutes, reason } => (
            "lanes.pause",
            json!({
                "scope": "neural_engine", "duration_ms": minutes * 60_000, "reason": reason,
            }),
        ),
        LanesCommand::Resume => ("lanes.resume", json!({})),
        LanesCommand::Status => ("lanes.status", json!({})),
    }
}

#[cfg(test)]
mod lane_command_tests {
    use super::*;

    #[test]
    fn pause_resume_status_encode_management_operations() {
        for (args, method, params) in [
            (
                vec![
                    "opctl",
                    "lanes",
                    "pause",
                    "--minutes",
                    "45",
                    "--reason",
                    "measurement",
                ],
                "lanes.pause",
                json!({"scope":"neural_engine","duration_ms":2_700_000,"reason":"measurement"}),
            ),
            (vec!["opctl", "lanes", "resume"], "lanes.resume", json!({})),
            (vec!["opctl", "lanes", "status"], "lanes.status", json!({})),
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            let Command::Lanes { command } = cli.command else {
                panic!("not a lanes command")
            };
            assert_eq!(lane_request(&command), (method, params));
        }
    }

    #[test]
    fn pause_minutes_have_a_hard_cli_cap() {
        for duration in ["0", "121", "18446744073709551615"] {
            assert!(Cli::try_parse_from([
                "opctl",
                "lanes",
                "pause",
                "--minutes",
                duration,
                "--reason",
                "measurement"
            ])
            .is_err());
        }
    }
}

#[derive(Debug, Subcommand)]
enum AdmissionCommand {
    /// Show the advisory scheduler snapshot.
    Status,
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
    /// Model id or fingerprint to require.
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
    let subc = discover(cli.subc.as_deref())
        .context("discover subc daemon connection file")?
        .path;
    let consumer = SubcConsumer::connect(&subc, ConsumerOptions::default())
        .await
        .with_context(|| format!("connect to subc through {}", subc.display()))?;
    let identity = BindIdentity::new("opctl", format!("opctl-{}", std::process::id()));

    let result = execute(&consumer, &identity, &cli.command, cli.json).await;
    consumer.close().await;
    result
}

async fn execute(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    command: &Command,
    raw_json: bool,
) -> Result<()> {
    let output = match command {
        Command::Lanes { command } => {
            let (method, params) = lane_request(command);
            call(consumer, identity, method, params).await
        }
        Command::Models { command } => match command {
            ModelsCommand::List => call(consumer, identity, "models.list", json!({})).await,
            ModelsCommand::Download { catalog_id } => {
                return model_download::run(consumer, identity, catalog_id, raw_json).await;
            }
            ModelsCommand::Remove { catalog_id } => {
                call(
                    consumer,
                    identity,
                    "models.remove",
                    json!({ "catalog_id": catalog_id }),
                )
                .await
            }
        },
        Command::Admission {
            command: AdmissionCommand::Status,
        } => call(consumer, identity, "admission.status", json!({})).await,
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
    }?;
    emit_output(&output, raw_json)?;
    ensure_no_error_envelope(&output)
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
    let timeout = Duration::from_secs(60);
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
    let disclosures = response
        .pointer("/result/truncation_disclosures")
        .and_then(Value::as_array);
    let mut seen = HashSet::with_capacity(vectors.len());
    for (index, item) in vectors.iter().enumerate() {
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
        let submitted_echo = item
            .get("submitted_sha256")
            .and_then(Value::as_str)
            .with_context(|| format!("embedding response item '{id}' omitted submitted_sha256"))?;
        let content_echo = item
            .get("content_sha256")
            .and_then(Value::as_str)
            .with_context(|| format!("embedding response item '{id}' omitted content_sha256"))?;

        verify_submitted_echo(text, submitted_echo)
            .with_context(|| format!("embedding response item '{id}' diverged"))?;

        if content_echo != submitted_echo {
            let disclosure = disclosures.and_then(|d| d.get(index));
            let is_truncated = disclosure
                .and_then(|d| d.get("truncated"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !is_truncated {
                bail!(
                    "embedding response item '{id}' diverged: content_sha256 differs from submitted_sha256 but truncation disclosure was not set"
                );
            }
            eprintln!(
                "embedding response item '{id}' truncated (submitted_sha256={submitted_echo}, content_sha256={content_echo})"
            );
        } else if let Some(disclosure) = disclosures.and_then(|d| d.get(index)) {
            let is_truncated = disclosure
                .get("truncated")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if is_truncated {
                bail!(
                    "embedding response item '{id}' diverged: truncation disclosure reported truncation but content_sha256 matches submitted_sha256"
                );
            }
        }

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

fn verify_submitted_echo(text: &str, echoed: &str) -> Result<String> {
    let expected = hex::encode(Sha256::digest(text.as_bytes()));
    if echoed != expected {
        bail!("submitted_sha256 mismatch: submitted={expected} echoed={echoed}");
    }
    Ok(expected)
}

#[cfg(test)]
fn verify_content_echo(text: &str, echoed: &str) -> Result<String> {
    let expected = hex::encode(Sha256::digest(text.as_bytes()));
    if echoed != expected {
        bail!("content_sha256 mismatch: submitted={expected} echoed={echoed}");
    }
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_models_download_with_global_json_flag() {
        let cli =
            Cli::try_parse_from(["opctl", "models", "download", "embedding", "--json"]).unwrap();
        assert!(cli.json);
        assert!(matches!(
            cli.command,
            Command::Models { command: ModelsCommand::Download { catalog_id } }
                if catalog_id == "embedding"
        ));
    }

    #[test]
    fn parses_models_remove() {
        let cli = Cli::try_parse_from(["opctl", "models", "remove", "embedding"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Models { command: ModelsCommand::Remove { catalog_id } }
                if catalog_id == "embedding"
        ));
    }

    #[test]
    fn verify_embedding_response_accepts_truncated_row_with_disclosure() {
        let submitted = vec![
            SubmittedItem {
                id: "w100".to_string(),
                text: "over cap long text".to_string(),
            },
            SubmittedItem {
                id: "w101".to_string(),
                text: "short".to_string(),
            },
        ];
        let w100_submitted_sha = hex::encode(Sha256::digest(b"over cap long text"));
        let w100_content_sha = hex::encode(Sha256::digest(b"over cap"));
        let w101_sha = hex::encode(Sha256::digest(b"short"));
        let response = json!({
            "result": {
                "fingerprint": "fp1",
                "table_epoch": 0,
                "dims": 2,
                "vectors": [
                    {
                        "id": "w100",
                        "vector": [1.0, 2.0],
                        "submitted_sha256": w100_submitted_sha,
                        "content_sha256": w100_content_sha,
                    },
                    {
                        "id": "w101",
                        "vector": [3.0, 4.0],
                        "submitted_sha256": w101_sha,
                        "content_sha256": w101_sha,
                    }
                ],
                "truncation_disclosures": [
                    {
                        "submitted_tokens": 10,
                        "effective_tokens": 5,
                        "truncated": true,
                    },
                    {
                        "submitted_tokens": 1,
                        "effective_tokens": 1,
                        "truncated": false,
                    }
                ]
            }
        });
        assert!(verify_embedding_response(&response, &submitted).is_ok());
    }

    #[test]
    fn verify_embedding_response_rejects_diverged_submitted_sha256() {
        let submitted = vec![SubmittedItem {
            id: "w100".to_string(),
            text: "real text".to_string(),
        }];
        let response = json!({
            "result": {
                "fingerprint": "fp1",
                "table_epoch": 0,
                "dims": 2,
                "vectors": [
                    {
                        "id": "w100",
                        "vector": [1.0, 2.0],
                        "submitted_sha256": "wrong_hash",
                        "content_sha256": "wrong_hash",
                    }
                ],
                "truncation_disclosures": [
                    {
                        "submitted_tokens": 2,
                        "effective_tokens": 2,
                        "truncated": false,
                    }
                ]
            }
        });
        let error = verify_embedding_response(&response, &submitted).unwrap_err();
        assert!(error.to_string().contains("diverged"));
    }

    #[test]
    fn verify_embedding_response_rejects_undisclosed_content_sha256_difference() {
        let submitted = vec![SubmittedItem {
            id: "w100".to_string(),
            text: "test text".to_string(),
        }];
        let submitted_sha = hex::encode(Sha256::digest(b"test text"));
        let response = json!({
            "result": {
                "fingerprint": "fp1",
                "table_epoch": 0,
                "dims": 2,
                "vectors": [
                    {
                        "id": "w100",
                        "vector": [1.0, 2.0],
                        "submitted_sha256": submitted_sha,
                        "content_sha256": "different_hash",
                    }
                ],
                "truncation_disclosures": [
                    {
                        "submitted_tokens": 2,
                        "effective_tokens": 2,
                        "truncated": false,
                    }
                ]
            }
        });
        let error = verify_embedding_response(&response, &submitted).unwrap_err();
        assert!(error.to_string().contains("diverged"));
    }

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
}
