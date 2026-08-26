#![forbid(unsafe_code)]

//! Generic management-surface caller for operator use: sends one method +
//! params JSON to any module over the fleet daemon and prints the raw
//! response envelope. Exists because module surfaces (approval asks,
//! campaign ops) are otherwise reachable only from inside module code.
//!
//! ## Chair-verb identity override
//!
//! Chair-only wire ops (for example alfonso-core `persona.seat_invite`)
//! authenticate the ROUTE identity against a recorded chair
//! (`harness` + `session`, exact bytes). The default ambient identity
//! (`subc-call` / `subc-call-<pid>`) is refused by construction.
//!
//! Pass `--identity <harness>:<session_id>` to stamp those exact bytes on
//! the consumer bind. When set, a loud disclosure line is printed to
//! stderr before the call so overridden calls are never mistaken for
//! ambient ones in logs or transcripts. The identity value is not logged
//! anywhere else.

use std::{env, io::Write, path::PathBuf, process, time::Duration};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use subc_client_rs::{CallOptions, ConsumerOptions, SubcConsumer};
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

const USAGE: &str = "\
usage: subc_call --module <id> --method <name> [--params <json>] [--subc <path>] [--identity <harness>:<session_id>]

  --module <id>       target module id (required)
  --method <name>     management-surface method (required)
  --params <json>     JSON object params (default: {})
  --subc <path>       subc connection file (default: $SYNAPSE_CONNECTION_FILE or XDG runtime path)
  --identity <h>:<s>  override bind harness:session (chair verbs; exact bytes)
  --help, -h          show this help
";

#[derive(Debug, PartialEq, Eq)]
struct CliArgs {
    subc: PathBuf,
    module: String,
    method: String,
    params: Value,
    /// When set, bind harness + session come from this override (exact bytes).
    identity_override: Option<(String, String)>,
}

#[derive(Debug, PartialEq, Eq)]
enum ParsedCli {
    Help,
    Run(CliArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    match parse_args(env::args().skip(1))? {
        ParsedCli::Help => {
            print!("{USAGE}");
            return Ok(());
        }
        ParsedCli::Run(args) => run(args).await,
    }
}

async fn run(args: CliArgs) -> Result<()> {
    if let Some((harness, session)) = &args.identity_override {
        // Loud disclosure only — identity must not appear in any other log path.
        writeln!(
            std::io::stderr(),
            "IDENTITY OVERRIDE: calling as {harness}:{session} (not the ambient module identity)"
        )
        .ok();
    }

    let consumer = SubcConsumer::connect(&args.subc, ConsumerOptions::default())
        .await
        .with_context(|| format!("connect to subc through {}", args.subc.display()))?;
    let identity = build_identity(
        args.identity_override.clone(),
        env::current_dir,
        process::id,
    )?;
    let response = consumer
        .call(
            RouteTarget::ManagementSurface {
                module_id: args.module.clone(),
            },
            identity,
            serde_json::to_vec(&json!({ "method": args.method, "params": args.params }))
                .context("encode request body")?,
            CallOptions {
                timeout: Duration::from_secs(120),
                ..CallOptions::default()
            },
        )
        .await
        .with_context(|| format!("call {} {}", args.module, args.method))?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    consumer.close().await;
    Ok(())
}

fn parse_args<I, S>(args: I) -> Result<ParsedCli>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut subc = default_subc_path();
    let mut module = None;
    let mut method = None;
    let mut params: Value = json!({});
    let mut identity_override = None;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        let arg = arg.as_ref();
        match arg {
            "--help" | "-h" => return Ok(ParsedCli::Help),
            "--subc" => subc = PathBuf::from(iter.next().context("--subc value")?.as_ref()),
            "--module" => {
                module = Some(iter.next().context("--module value")?.as_ref().to_string())
            }
            "--method" => {
                method = Some(iter.next().context("--method value")?.as_ref().to_string())
            }
            "--params" => {
                let raw = iter.next().context("--params value")?;
                params =
                    serde_json::from_str(raw.as_ref()).context("--params must be a JSON object")?;
            }
            "--identity" => {
                let raw = iter.next().context("--identity value")?;
                identity_override = Some(parse_identity_override(raw.as_ref())?);
            }
            other => bail!("unknown argument: {other}\n\n{USAGE}"),
        }
    }
    let module = module.context(format!("--module is required\n\n{USAGE}"))?;
    let method = method.context(format!("--method is required\n\n{USAGE}"))?;
    Ok(ParsedCli::Run(CliArgs {
        subc,
        module,
        method,
        params,
        identity_override,
    }))
}

/// Split `--identity <harness>:<session_id>` on the first `:`.
/// Both sides must be non-empty; harness/session pass through exactly.
fn parse_identity_override(raw: &str) -> Result<(String, String)> {
    let (harness, session) = raw
        .split_once(':')
        .with_context(|| format!("--identity must be <harness>:<session_id>, got {raw:?}"))?;
    if harness.is_empty() || session.is_empty() {
        bail!("--identity must be <harness>:<session_id> with non-empty parts, got {raw:?}");
    }
    Ok((harness.to_string(), session.to_string()))
}

/// Build the bind identity. Override path uses exact harness/session bytes;
/// default path matches the historical ambient `subc-call` / `subc-call-<pid>`.
fn build_identity<Cwd, Pid>(
    identity_override: Option<(String, String)>,
    cwd: Cwd,
    pid: Pid,
) -> Result<BindIdentity>
where
    Cwd: FnOnce() -> std::io::Result<PathBuf>,
    Pid: FnOnce() -> u32,
{
    let project_root = cwd().context("resolve cwd")?;
    match identity_override {
        Some((harness, session)) => Ok(BindIdentity {
            project_root,
            harness,
            session,
        }),
        None => Ok(BindIdentity {
            project_root,
            harness: "subc-call".to_string(),
            session: format!("subc-call-{}", pid()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn unwrap_run(parsed: ParsedCli) -> CliArgs {
        match parsed {
            ParsedCli::Run(args) => args,
            ParsedCli::Help => panic!("expected Run, got Help"),
        }
    }

    #[test]
    fn parse_requires_module_and_method() {
        let err = parse_args(["--module", "m"]).unwrap_err();
        assert!(err.to_string().contains("--method is required"));
        let err = parse_args(["--method", "x"]).unwrap_err();
        assert!(err.to_string().contains("--module is required"));
    }

    #[test]
    fn parse_default_has_no_identity_override() {
        let args = unwrap_run(parse_args(["--module", "synapse", "--method", "ping"]).unwrap());
        assert_eq!(args.module, "synapse");
        assert_eq!(args.method, "ping");
        assert_eq!(args.params, json!({}));
        assert_eq!(args.identity_override, None);
        assert_eq!(args.subc, PathBuf::from(DEFAULT_SUBC));
    }

    #[test]
    fn parse_identity_override_exact_bytes() {
        let args = unwrap_run(
            parse_args([
                "--module",
                "alfonso-core",
                "--method",
                "persona.seat_invite",
                "--identity",
                "alfonso:sess-abc:with:colons",
            ])
            .unwrap(),
        );
        assert_eq!(
            args.identity_override
                .as_ref()
                .map(|(h, s)| (h.as_str(), s.as_str())),
            Some(("alfonso", "sess-abc:with:colons"))
        );
    }

    #[test]
    fn parse_identity_rejects_missing_colon_and_empty_parts() {
        let err =
            parse_args(["--module", "m", "--method", "x", "--identity", "no-colon"]).unwrap_err();
        assert!(err.to_string().contains("--identity must be"));

        let err =
            parse_args(["--module", "m", "--method", "x", "--identity", ":session"]).unwrap_err();
        assert!(err.to_string().contains("non-empty parts"));

        let err =
            parse_args(["--module", "m", "--method", "x", "--identity", "harness:"]).unwrap_err();
        assert!(err.to_string().contains("non-empty parts"));
    }

    #[test]
    fn parse_help() {
        assert_eq!(parse_args(["--help"]).unwrap(), ParsedCli::Help);
        assert_eq!(parse_args(["-h"]).unwrap(), ParsedCli::Help);
        assert!(USAGE.contains("--identity"));
        assert!(USAGE.contains("usage: subc_call"));
    }

    #[test]
    fn build_identity_default_matches_ambient() {
        let id = build_identity(None, || Ok(PathBuf::from("/tmp/proj")), || 4242).unwrap();
        assert_eq!(id.project_root, PathBuf::from("/tmp/proj"));
        assert_eq!(id.harness, "subc-call");
        assert_eq!(id.session, "subc-call-4242");
    }

    #[test]
    fn build_identity_override_passes_exact_bytes() {
        let id = build_identity(
            Some(("chair-harness".into(), "seat-session-01".into())),
            || Ok(PathBuf::from("/work")),
            || 1,
        )
        .unwrap();
        assert_eq!(id.project_root, PathBuf::from("/work"));
        assert_eq!(id.harness, "chair-harness");
        assert_eq!(id.session, "seat-session-01");
    }

    #[test]
    fn parse_identity_override_unit() {
        assert_eq!(
            parse_identity_override("h:s").unwrap(),
            ("h".into(), "s".into())
        );
        assert_eq!(
            parse_identity_override("h:s:extra").unwrap(),
            ("h".into(), "s:extra".into())
        );
    }
}
