use std::{path::PathBuf, sync::Arc, time::Duration};

use cortexkit_bus_trait::{ContentDigest, Headers};
use rusqlite::{params, OptionalExtension};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::store::{Event, SynapseStore};

const FINDINGS_SUBJECT: &str = "event.synapse.findings_changed.v1";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GetParams {
    event_key: String,
    event_name: String,
    version: u32,
}

pub(super) fn events_get(
    state: &crate::ModuleState,
    params: Value,
) -> subc_client_rs::HandlerOutcome {
    let params: GetParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(_) => {
            return crate::channel_error(
                "invalid_request",
                "events_get requires event_key, event_name and version",
            )
        }
    };
    if params.event_name != "findings_changed" || params.version != 1 {
        return crate::channel_error("event_not_visible", "event is not visible");
    }
    let row = state.store.sqlite().with_conn(|conn| {
        conn.query_row(
            "SELECT body,created_at FROM event_outbox WHERE event_id=?1 AND subject=?2",
            params![params.event_key, FINDINGS_SUBJECT],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
        )
        .optional()
    });
    match row {
        Ok(Some((body, created_at))) => {
            let Some(timestamp) = chrono::DateTime::from_timestamp_millis(created_at) else {
                return crate::channel_error("store_failure", "event timestamp is invalid");
            };
            crate::result_outcome(
                json!({"body":body,"digest":ContentDigest::of_bytes(body.as_bytes()).to_string(),
                "event_key":params.event_key,"event_name":params.event_name,
                "published_at":timestamp.to_rfc3339_opts(chrono::SecondsFormat::Millis,true)}),
            )
        }
        Ok(None) => crate::channel_error("event_not_visible", "event is not visible"),
        Err(error) => crate::channel_error("store_failure", error.to_string()),
    }
}

enum Content {
    Inline,
    ClaimCheck {
        digest: ContentDigest,
        headers: Headers,
    },
}

fn content(event: &Event) -> Content {
    if event.subject == FINDINGS_SUBJECT {
        Content::ClaimCheck {
            digest: ContentDigest::of_bytes(event.body.as_bytes()),
            headers: Headers::from([("event_key".into(), event.event_id.clone())]),
        }
    } else {
        Content::Inline
    }
}

trait EventBus {
    async fn publish(&mut self, event: &Event, content: Content) -> Result<bool, String>;
}

struct FleetBus {
    handle: subc_client_rs::ModuleHandle,
    connection_file: PathBuf,
    client: Option<cortexkit_bus_client::BusClient>,
    module_id: String,
}

impl EventBus for FleetBus {
    async fn publish(&mut self, event: &Event, content: Content) -> Result<bool, String> {
        if self
            .client
            .as_ref()
            .is_some_and(|client| client.connection_changes().has_changed().is_err())
        {
            self.client = None;
        }
        if self.client.is_none() {
            let consumer = self
                .handle
                .consumer(&self.connection_file, Default::default())
                .await
                .map_err(|e| e.to_string())?;
            let options = cortexkit_bus_client::BusClientOptions::new(&self.module_id)
                .map_err(|e| e.to_string())?;
            self.client = Some(
                cortexkit_bus_client::BusClient::connect(consumer, options)
                    .await
                    .map_err(|e| e.to_string())?,
            );
        }
        let name = event
            .subject
            .strip_prefix("event.synapse.")
            .and_then(|s| s.strip_suffix(".v1"))
            .ok_or("invalid outbox subject")?;
        let client = self.client.as_ref().expect("connected client");
        let receipt = match content {
            Content::Inline => client
                .publish_event(name, 1, None, &event.event_id, event.body.clone())
                .await
                .map_err(|e| e.to_string())?,
            Content::ClaimCheck { digest, headers } => client
                .publish_claim_check(
                    "findings_changed",
                    1,
                    None,
                    &event.event_id,
                    digest,
                    headers,
                )
                .await
                .map_err(|e| e.to_string())?,
        };
        Ok(receipt.duplicate())
    }
}

struct Clock {
    epoch_ms: u64,
    started: tokio::time::Instant,
}

impl Clock {
    fn new(epoch_ms: u64) -> Self {
        Self {
            epoch_ms,
            started: tokio::time::Instant::now(),
        }
    }
    fn now(&self) -> u64 {
        self.epoch_ms
            .saturating_add(self.started.elapsed().as_millis() as u64)
    }
}

// Rows reach the outbox inside the fact writers' own store transactions (migration
// 18's triggers, and embed page commits), so nothing wakes the publisher directly. It
// polls one indexed row once a second when idle and drains back to back while rows
// wait. Pruning runs at most once a minute, so an idle module does no store writes.
const IDLE_POLL: Duration = Duration::from_secs(1);
const PRUNE_EVERY_MS: u64 = 60_000;

async fn drain(store: Arc<SynapseStore>, mut bus: impl EventBus, clock: Clock) {
    let mut outage = false;
    let mut backoff = Duration::from_millis(250);
    let mut last_prune: Option<u64> = None;
    loop {
        let result = async {
            let now = clock.now();
            if last_prune.is_none_or(|at| now.saturating_sub(at) >= PRUNE_EVERY_MS) {
                store.prune_events(now).map_err(|e| e.to_string())?;
                last_prune = Some(now);
            }
            if let Some(event) = store.next_event().map_err(|e| e.to_string())? {
                let duplicate = tokio::time::timeout(
                    Duration::from_secs(5),
                    bus.publish(&event, content(&event)),
                )
                .await
                .map_err(|_| "bus publish timed out".to_string())??;
                tracing::debug!(event_id=%event.event_id, duplicate, "outbox event acknowledged");
                store
                    .mark_event_published(&event.event_id, clock.now())
                    .map_err(|e| e.to_string())?;
                Ok(true)
            } else {
                Ok::<_, String>(false)
            }
        }
        .await;
        match result {
            Ok(published) => {
                if published && outage {
                    tracing::info!(target: "lifecycle", "event publisher reconnected");
                    outage = false;
                }
                backoff = Duration::from_millis(250);
                if !published {
                    tokio::time::sleep(IDLE_POLL).await;
                }
            }
            Err(error) => {
                if !outage {
                    tracing::warn!(
                        target: "lifecycle",
                        %error,
                        "event publisher unavailable; committed events remain in the outbox"
                    );
                    outage = true;
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(30));
            }
        }
    }
}

pub(super) fn start(
    state: &crate::ModuleState,
    handle: subc_client_rs::ModuleHandle,
    connection_file: PathBuf,
) {
    let bus = FleetBus {
        handle,
        connection_file,
        client: None,
        module_id: state.module_id.clone(),
    };
    tracing::info!(target: "lifecycle", "event publisher started");
    tokio::spawn(drain(state.store.clone(), bus, Clock::new(crate::now_ms())));
}

#[cfg(test)]
#[path = "event_outbox_tests.rs"]
mod tests;
