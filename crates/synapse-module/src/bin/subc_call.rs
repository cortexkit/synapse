#![forbid(unsafe_code)]

//! Generic management-surface caller for operator use: sends one method +
//! params JSON to any module over the fleet daemon and prints the raw
//! response envelope. Exists because module surfaces (approval asks,
//! campaign ops) are otherwise reachable only from inside module code.

use std::{env, path::PathBuf, time::Duration};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use subc_client_rs::{CallOptions, ConsumerOptions, SubcConsumer};
use subc_protocol::{BindIdentity, RouteTarget};

const DEFAULT_SUBC: &str = "/Users/[owner]/.local/share/cortexkit/run/subc-connection.json";

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let mut subc = PathBuf::from(DEFAULT_SUBC);
    let mut module = None;
    let mut method = None;
    let mut params: Value = json!({});
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--subc" => subc = PathBuf::from(args.next().context("--subc value")?),
            "--module" => module = Some(args.next().context("--module value")?),
            "--method" => method = Some(args.next().context("--method value")?),
            "--params" => {
                params = serde_json::from_str(&args.next().context("--params value")?)
                    .context("--params must be a JSON object")?
            }
            other => bail!("unknown argument: {other}"),
        }
    }
    let module = module.context("--module is required")?;
    let method = method.context("--method is required")?;

    let consumer = SubcConsumer::connect(&subc, ConsumerOptions::default())
        .await
        .with_context(|| format!("connect to subc through {}", subc.display()))?;
    let identity = BindIdentity {
        project_root: env::current_dir().context("resolve cwd")?,
        harness: "subc-call".to_string(),
        session: format!("subc-call-{}", std::process::id()),
    };
    let response = consumer
        .call(
            RouteTarget::ManagementSurface {
                module_id: module.clone(),
            },
            identity,
            serde_json::to_vec(&json!({ "method": method, "params": params }))
                .context("encode request body")?,
            CallOptions {
                timeout: Duration::from_secs(120),
                ..CallOptions::default()
            },
        )
        .await
        .with_context(|| format!("call {module} {method}"))?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    consumer.close().await;
    Ok(())
}
