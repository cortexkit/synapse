use std::{ops::AsyncFnMut, time::Duration};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use subc_client_rs::SubcConsumer;
use subc_protocol::BindIdentity;

use crate::{call, emit_output, ensure_no_error_envelope, response_error, CommandOutput};

const POLL_INTERVAL: Duration = Duration::from_secs(1);

pub(super) async fn run(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    catalog_id: &str,
    raw_json: bool,
) -> Result<()> {
    let download = async {
        let mut last_progress = String::new();
        let (output, outcome) = download_and_watch(
            catalog_id,
            async |method, params| call(consumer, identity, method, params).await,
            POLL_INTERVAL,
            |status| {
                if !raw_json {
                    let line = progress_line(status);
                    if line != last_progress {
                        eprintln!("{catalog_id}: {line}");
                        last_progress = line;
                    }
                }
            },
        )
        .await?;

        if raw_json {
            emit_output(&output, true)?;
        }
        match outcome {
            Outcome::Committed => {
                if !raw_json {
                    let lanes = installed_lanes(consumer, identity, catalog_id).await;
                    match lanes {
                        Ok(lanes) => println!("{catalog_id} is installed; lanes: {lanes}"),
                        Err(error) => {
                            println!("{catalog_id} is installed; could not list lanes: {error:#}")
                        }
                    }
                }
                Ok(())
            }
            Outcome::Failed => {
                let status = &output.primary()["result"];
                let code = status["error"]["code"].as_str().unwrap_or("unknown");
                let message = status["error"]["message"]
                    .as_str()
                    .unwrap_or("no failure message returned");
                bail!("{catalog_id}: download failed (code={code}): {message}");
            }
            Outcome::Cancelled => bail!("{catalog_id}: download cancelled"),
            Outcome::Refused => ensure_no_error_envelope(&output),
        }
    };

    tokio::select! {
        biased;
        signal = tokio::signal::ctrl_c() => {
            signal.context("listen for Ctrl-C")?;
            bail!("stopped watching {catalog_id}. An accepted download continues in the background. \
                   Run `ck-synapse-opctl models download {catalog_id}` again to resume watching.");
        }
        result = download => result,
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Committed,
    Failed,
    Cancelled,
    Refused,
}

async fn download_and_watch(
    catalog_id: &str,
    mut request: impl AsyncFnMut(&'static str, Value) -> Result<CommandOutput>,
    interval: Duration,
    mut progress: impl FnMut(&Value),
) -> Result<(CommandOutput, Outcome)> {
    // Let Synapse derive the key from the catalog ID and manifest. Rerunning
    // attaches to an active job or reuses a commit if the model is still installed.
    let mut output = request("models.download", json!({ "catalog_id": catalog_id })).await?;
    if response_error(output.primary()).is_some() {
        return Ok((output, Outcome::Refused));
    }
    let job_id = output.primary()["result"]["job_id"]
        .as_str()
        .context("models.download reply omitted job_id")?
        .to_owned();

    loop {
        if response_error(output.primary()).is_some() {
            return Ok((output, Outcome::Refused));
        }
        let status = &output.primary()["result"];
        let outcome = match status["state"].as_str() {
            Some("committed") => Outcome::Committed,
            Some("failed") => Outcome::Failed,
            Some("cancelled") => Outcome::Cancelled,
            Some("queued" | "downloading" | "verifying") => {
                progress(status);
                tokio::time::sleep(interval).await;
                output = request("model.status", json!({ "job_id": job_id })).await?;
                continue;
            }
            state => bail!("download job {job_id} returned an unrecognized state: {state:?}"),
        };
        return Ok((output, outcome));
    }
}

fn progress_line(status: &Value) -> String {
    let state = status["state"].as_str().unwrap_or_default();
    match (
        status["bytes_done"].as_u64(),
        status["bytes_total"].as_u64(),
    ) {
        (Some(done), Some(total)) => format!("{state} ({done}/{total} bytes)"),
        _ => state.to_owned(),
    }
}

async fn installed_lanes(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    catalog_id: &str,
) -> Result<String> {
    // Job status has no lane IDs; the catalog reports current installs per backend.
    let catalog = call(
        consumer,
        identity,
        "models.catalog",
        json!({ "query": catalog_id }),
    )
    .await?;
    ensure_no_error_envelope(&catalog)?;
    let backends = catalog
        .primary()
        .pointer("/result/models")
        .and_then(Value::as_array)
        .and_then(|models| models.iter().find(|model| model["id"] == catalog_id))
        .and_then(|model| model["backends"].as_array())
        .context("catalog reply omitted the requested model's backends")?;
    let lanes: Vec<_> = backends
        .iter()
        .filter(|backend| backend["installed"] == true && backend["runnable"] == true)
        .filter_map(|backend| backend["lane_id"].as_str())
        .collect();
    if lanes.is_empty() {
        bail!("no runnable installed lanes reported by the catalog");
    }
    Ok(lanes.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn watches_model_download_until_commit_and_accepts_committed_replay() {
        let catalog_id = "gte-modernbert-base";
        let job_id = "download-job-1";
        let status_params = json!({ "job_id": job_id });
        let download_params = json!({ "catalog_id": catalog_id });
        let committed =
            json!({ "kind": "models.download", "job_id": job_id, "state": "committed" });
        let replay = json!({ "job_id": job_id, "state": "committed" });
        let mut replies = [
            ("models.download", download_params.clone(), json!({ "job_id": job_id, "state": "queued" })),
            ("model.status", status_params.clone(), json!({ "kind": "models.download", "job_id": job_id, "state": "downloading", "bytes_done": 2, "bytes_total": 3 })),
            ("model.status", status_params.clone(), json!({ "kind": "models.download", "job_id": job_id, "state": "verifying" })),
            ("model.status", status_params, committed.clone()),
            ("models.download", download_params, replay.clone()),
        ]
        .into_iter();
        let mut request = async |method, params| {
            let (expected_method, expected_params, reply) =
                replies.next().expect("unexpected extra call");
            assert_eq!(method, expected_method);
            assert_eq!(params, expected_params);
            Ok(CommandOutput::one(json!({ "result": reply })))
        };
        let mut progress = Vec::new();

        let (output, outcome) =
            download_and_watch(catalog_id, &mut request, Duration::ZERO, |status| {
                progress.push(progress_line(status))
            })
            .await
            .unwrap();
        assert_eq!(outcome, Outcome::Committed);
        assert_eq!(output.primary(), &json!({ "result": committed }));
        assert_eq!(progress, ["queued", "downloading (2/3 bytes)", "verifying"]);

        let (output, outcome) =
            download_and_watch(catalog_id, &mut request, Duration::ZERO, |_| {
                panic!("committed replay should not emit progress")
            })
            .await
            .unwrap();
        assert_eq!(outcome, Outcome::Committed);
        assert_eq!(output.primary(), &json!({ "result": replay }));
        assert!(replies.next().is_none());
    }
}
