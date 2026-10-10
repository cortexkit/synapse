//! Public catalog protocol, including bounded cold-load retries before grading.
use crate::refuse;
use serde_json::{json, Value};
use std::time::Duration;
use synapse_certify::Result;
use tokio::time::{timeout_at, Instant};

pub(crate) const RUN_BUDGET: Duration = Duration::from_secs(3600);
pub(crate) trait Client {
    async fn request(&mut self, method: &str, params: Value) -> Result<Value>;
}
pub(crate) struct Pin {
    pub lane_id: String,
    pub fingerprint: String,
    pub task: String,
}

async fn bounded(
    client: &mut impl Client,
    method: &str,
    params: Value,
    deadline: Instant,
) -> Result<Value> {
    if Instant::now() >= deadline {
        return Err(refuse("runner deadline exceeded"));
    }
    timeout_at(deadline, client.request(method, params))
        .await
        .map_err(|_| refuse("runner deadline exceeded"))?
}
fn success(value: Value) -> Result<Value> {
    if value.get("error").is_some() {
        return Err(refuse(format!("candidate error: {value}")));
    }
    Ok(value)
}

pub(crate) async fn install(
    client: &mut impl Client,
    row: &str,
    model: &str,
    deadline: Instant,
) -> Result<Pin> {
    let backend =
        synapse_certify::row_backend(row).ok_or_else(|| refuse("unknown certification row"))?;
    let catalog = success(bounded(client, "models.catalog", json!({}), deadline).await?)?;
    let entry = catalog["models"]
        .as_array()
        .and_then(|models| models.iter().find(|entry| entry["id"] == model))
        .ok_or_else(|| refuse("catalog cell undeclared"))?;
    let lane = entry["backends"]
        .as_array()
        .and_then(|lanes| lanes.iter().find(|lane| lane["backend"] == backend))
        .ok_or_else(|| refuse("catalog cell undeclared"))?;
    if lane["runnable"] != true {
        return Err(refuse(format!("catalog lane not runnable: {lane}")));
    }
    let lane_id = format!("{model}-{backend}");
    if lane["lane_id"] != lane_id {
        return Err(refuse("catalog lane id mismatch"));
    }
    let pin = Pin {
        lane_id,
        fingerprint: lane["fingerprint"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| refuse("catalog pin missing"))?
            .into(),
        task: entry["task"]
            .as_str()
            .filter(|s| matches!(*s, "embed" | "rerank"))
            .ok_or_else(|| refuse("catalog task missing"))?
            .into(),
    };
    let admitted = success(
        bounded(
            client,
            "models.download",
            json!({"catalog_id": model}),
            deadline,
        )
        .await?,
    )?;
    let job = admitted["job_id"]
        .as_str()
        .ok_or_else(|| refuse("download job id missing"))?;
    loop {
        let status = bounded(client, "model.status", json!({"job_id": job}), deadline).await?;
        match status["state"].as_str() {
            Some("committed") => {
                success(status)?;
                return Ok(pin);
            }
            Some("failed" | "cancelled") => {
                return Err(refuse(format!("catalog download refused: {status}")))
            }
            Some("queued" | "downloading") => {
                success(status)?;
            }
            _ => return Err(refuse(format!("invalid download status: {status}"))),
        }
        timeout_at(deadline, tokio::time::sleep(Duration::from_millis(100)))
            .await
            .map_err(|_| refuse("runner deadline exceeded"))?;
    }
}

pub(crate) struct CertifiedSession<'a, C> {
    pub client: &'a mut C,
    pub pin: Pin,
    deadline: Instant,
    checked: bool,
}
impl<'a, C: Client> CertifiedSession<'a, C> {
    pub fn new(client: &'a mut C, pin: Pin, deadline: Instant) -> Self {
        Self {
            client,
            pin,
            deadline,
            checked: false,
        }
    }
    pub async fn plain(&mut self, method: &str, params: Value) -> Result<Value> {
        success(bounded(self.client, method, params, self.deadline).await?)
    }
    pub async fn observations(&mut self) -> Result<Value> {
        let value = self.plain("certify.observations", json!({})).await?;
        if value["available"] != true {
            return Err(refuse(format!("certify.observations unavailable: {value}")));
        }
        Ok(value)
    }
    pub fn check_fingerprint(&self, response: &Value) -> Result<()> {
        if response["fingerprint"] != self.pin.fingerprint {
            return Err(refuse("response fingerprint differs from catalog pin"));
        }
        Ok(())
    }
    pub async fn fixture(&mut self, method: &str, mut params: Value) -> Result<Value> {
        params["model"] = json!(self.pin.lane_id);
        params["deadline_ms"] = json!(RUN_BUDGET.as_millis() as u64);
        loop {
            let response = bounded(self.client, method, params.clone(), self.deadline).await?;
            if response["error"]["code"] == "model_loading" {
                let hint = response["error"]["retry_after_ms"]
                    .as_u64()
                    .unwrap_or(100)
                    .max(1);
                timeout_at(
                    self.deadline,
                    tokio::time::sleep(Duration::from_millis(hint)),
                )
                .await
                .map_err(|_| refuse("runner deadline exceeded"))?;
                continue;
            }
            let response = success(response)?;
            self.check_fingerprint(&response)?;
            if !self.checked {
                let listed = self.plain("models.list", json!({})).await?;
                let lane = listed["models"]
                    .as_array()
                    .and_then(|models| {
                        models
                            .iter()
                            .find(|row| row["model_id"] == self.pin.lane_id)
                    })
                    .ok_or_else(|| refuse("loaded catalog lane missing from models.list"))?;
                if lane["self_check"]["state"] != "passed"
                    || lane["fingerprints"] != json!([self.pin.fingerprint])
                {
                    return Err(refuse(format!(
                        "catalog lane check or pin mismatch: {lane}"
                    )));
                }
                self.checked = true;
            }
            return Ok(response);
        }
    }
    pub async fn oversized(&mut self, method: &str, mut params: Value) -> Result<Value> {
        params["model"] = json!(self.pin.lane_id);
        params["deadline_ms"] = json!(RUN_BUDGET.as_millis() as u64);
        let response = bounded(self.client, method, params, self.deadline).await?;
        if response.get("error").is_none() {
            self.check_fingerprint(&response)?;
        }
        if response["error"]["code"] != "sequence_too_long" {
            return Err(refuse(format!("8193 admission refused: {response}")));
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    struct Stub {
        replies: VecDeque<Value>,
        calls: Vec<(String, Value, Instant)>,
    }
    impl Client for Stub {
        async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
            self.calls.push((method.into(), params, Instant::now()));
            Ok(self.replies.pop_front().expect("unexpected request"))
        }
    }
    fn stub(replies: Vec<Value>) -> Stub {
        Stub {
            replies: replies.into(),
            calls: vec![],
        }
    }
    fn pin() -> Pin {
        Pin {
            lane_id: "model-metal".into(),
            fingerprint: "pin".into(),
            task: "embed".into(),
        }
    }
    fn listed() -> Value {
        json!({"models": [{"model_id": "model-metal", "self_check": {"state": "passed"}, "fingerprints": ["pin"]}]})
    }
    #[tokio::test]
    async fn loading_retries_same_fixture_after_hint_and_never_completes_it() {
        let mut client = stub(vec![
            json!({"error":{"code":"model_loading","retry_after_ms":10}}),
            json!({"error":{"code":"model_loading","retry_after_ms":10}}),
            json!({"fingerprint":"pin","fixture":"first"}),
            listed(),
            json!({"fingerprint":"pin","fixture":"second"}),
        ]);
        let mut session =
            CertifiedSession::new(&mut client, pin(), Instant::now() + Duration::from_secs(2));
        let mut completed = Vec::new();
        for _ in 0..2 {
            completed.push(
                session
                    .fixture("embed.query", json!({"text":"fixture"}))
                    .await
                    .unwrap()["fixture"]
                    .clone(),
            );
        }
        assert_eq!(completed, [json!("first"), json!("second")]);
        assert_eq!(
            client
                .calls
                .iter()
                .map(|c| c.0.as_str())
                .collect::<Vec<_>>(),
            [
                "embed.query",
                "embed.query",
                "embed.query",
                "models.list",
                "embed.query"
            ]
        );
        assert_eq!(client.calls[0].1, client.calls[1].1);
        assert_eq!(client.calls[1].1, client.calls[2].1);
        assert_eq!(client.calls[0].1["model"], "model-metal");
        assert!(client.calls[0].1["deadline_ms"].as_u64().unwrap() > 5000);
        for n in 1..3 {
            assert!(
                client.calls[n].2.duration_since(client.calls[n - 1].2)
                    >= Duration::from_millis(10)
            );
        }
    }
    #[tokio::test]
    async fn loading_deadline_refuses_without_completing_a_fixture() {
        let mut client = stub(vec![
            json!({"error":{"code":"model_loading","retry_after_ms":100}}),
        ]);
        let mut session = CertifiedSession::new(
            &mut client,
            pin(),
            Instant::now() + Duration::from_millis(10),
        );
        assert!(session
            .fixture("embed.query", json!({}))
            .await
            .unwrap_err()
            .to_string()
            .contains("runner deadline"));
        assert_eq!(client.calls.len(), 1);
        let mut session = CertifiedSession::new(&mut client, pin(), Instant::now());
        assert!(session
            .fixture("embed.query", json!({}))
            .await
            .unwrap_err()
            .to_string()
            .contains("runner deadline"));
        assert_eq!(client.calls.len(), 1);
    }

    #[tokio::test]
    async fn unanswered_fixture_is_bounded_by_runner_deadline() {
        struct Silent;
        impl Client for Silent {
            async fn request(&mut self, _: &str, _: Value) -> Result<Value> {
                std::future::pending().await
            }
        }
        let mut client = Silent;
        let mut session = CertifiedSession::new(
            &mut client,
            pin(),
            Instant::now() + Duration::from_millis(10),
        );
        assert!(session
            .fixture("embed.query", json!({}))
            .await
            .unwrap_err()
            .to_string()
            .contains("runner deadline"));
    }

    #[tokio::test]
    async fn oversized_success_after_truncation_cannot_be_an_admission_pass() {
        let mut client = stub(vec![
            json!({"fingerprint":"pin","truncation_disclosures":[{"submitted_tokens":8193,"effective_tokens":8192,"truncated":true}]}),
        ]);
        let mut session =
            CertifiedSession::new(&mut client, pin(), Instant::now() + Duration::from_secs(1));
        assert!(session
            .oversized("embed.query", json!({"text":"oversized"}))
            .await
            .unwrap_err()
            .to_string()
            .contains("8193 admission refused"));
        assert_eq!(client.calls.len(), 1);
    }
    #[tokio::test]
    async fn fatal_codes_refuse_instead_of_completing_or_retrying() {
        for code in [
            "artifact_invalid",
            "catalog_self_check_failed",
            "self_check_failed",
            "model_not_installed",
            "engine_crashed",
            "deadline_exceeded",
            "unknown_future_code",
        ] {
            let mut client = stub(vec![
                json!({"error":{"code":code,"message":"self-check exceeded 2000 ms","retry_after_ms":1}}),
            ]);
            let mut session =
                CertifiedSession::new(&mut client, pin(), Instant::now() + Duration::from_secs(1));
            assert!(session
                .fixture("embed.query", json!({}))
                .await
                .unwrap_err()
                .to_string()
                .contains(code));
            assert_eq!(client.calls.len(), 1);
        }
    }
    #[tokio::test]
    async fn every_success_must_carry_the_pin_and_list_must_show_passed() {
        for bad in [json!({}), json!({"fingerprint":"wrong"})] {
            let mut client = stub(vec![json!({"fingerprint":"pin"}), listed(), bad]);
            let mut session =
                CertifiedSession::new(&mut client, pin(), Instant::now() + Duration::from_secs(1));
            session.fixture("embed.query", json!({})).await.unwrap();
            assert!(session
                .fixture("embed.query", json!({}))
                .await
                .unwrap_err()
                .to_string()
                .contains("fingerprint"));
        }
        for (key, value) in [
            ("self_check", json!({"state":"running"})),
            ("fingerprints", json!(["wrong"])),
            ("fingerprints", json!(["pin", "pin"])),
        ] {
            let mut bad = listed();
            bad["models"][0][key] = value;
            let mut client = stub(vec![json!({"fingerprint":"pin"}), bad]);
            let mut session =
                CertifiedSession::new(&mut client, pin(), Instant::now() + Duration::from_secs(1));
            assert!(session
                .fixture("embed.query", json!({}))
                .await
                .unwrap_err()
                .to_string()
                .contains("check or pin mismatch"));
        }
    }
    #[tokio::test]
    async fn checks_catalog_before_download_and_requires_committed_job() {
        let catalog = json!({"models":[{"id":"model","task":"embed","backends":[{"backend":"metal","lane_id":"model-metal","fingerprint":"pin","runnable":true}]}]});
        let mut client = stub(vec![
            catalog.clone(),
            json!({"job_id":"job","state":"queued"}),
            json!({"state":"downloading"}),
            json!({"state":"committed"}),
        ]);
        let got = install(
            &mut client,
            "metal-m5",
            "model",
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap();
        assert_eq!(got.fingerprint, "pin");
        assert_eq!(client.calls[1].1, json!({"catalog_id":"model"}));
        assert_eq!(client.calls[2].1, json!({"job_id":"job"}));
        for state in ["failed", "cancelled"] {
            let mut client = stub(vec![
                catalog.clone(),
                json!({"job_id":"job"}),
                json!({"state":state}),
            ]);
            assert!(install(
                &mut client,
                "metal-m5",
                "model",
                Instant::now() + Duration::from_secs(1)
            )
            .await
            .is_err());
        }
        let mut bad = catalog;
        bad["models"][0]["backends"][0]["runnable"] = json!(false);
        let mut client = stub(vec![bad]);
        assert!(install(
            &mut client,
            "metal-m5",
            "model",
            Instant::now() + Duration::from_secs(1)
        )
        .await
        .is_err());
        assert_eq!(client.calls.len(), 1);
    }
}
