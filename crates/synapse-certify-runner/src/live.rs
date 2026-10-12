//! Executes candidate artifacts against a private in-process daemon.
use crate::refuse;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};
use subc_daemon::{
    daemon_config::StorageConfig, serve_listener, ControlHandler, Registry, Router, ServerAuth,
};
use subc_protocol::{BindIdentity, Flags, Frame, FrameType, Priority, RouteTarget};
use subc_transport::{
    authenticate_client, generate_daemon_id, generate_key, read_frame, write_atomic, write_frame,
    ConnectionInfo, Endpoint, SCHEMA_VERSION,
};
use synapse_certify::{Admission, Inventory, Outcome, Parity, Result, RunEvidence, Runner};
use synapse_parity::evaluator::{
    evaluate, rerank_tolerance, Case, Evaluation, FixtureSet, ObservedCase, Output,
};
use tokio::{
    net::{TcpListener, TcpStream},
    process::Command,
    time::timeout,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub assets: PathBuf,
    pub checkout: PathBuf,
    pub mirror: PathBuf,
}

pub struct LiveRunner {
    options: Options,
    source: String,
    runtime: tokio::runtime::Runtime,
    floor: Option<Value>,
    layout: Option<(String, crate::assets::Layout)>,
}
fn err(error: impl std::fmt::Display) -> crate::Error {
    refuse(error.to_string())
}
pub(crate) fn lane(row: &str) -> &'static str {
    if row == "metal-m5" {
        "owned-metal"
    } else if row == "ane-m5" {
        "ane-direct-worker"
    } else if row.starts_with("cuda-") {
        "owned-cuda"
    } else {
        "owned-vulkan"
    }
}
pub(crate) fn worker(row: &str) -> Option<&'static str> {
    match lane(row) {
        "owned-cuda" => Some("ck-synapse-worker-cuda"),
        "owned-vulkan" => Some("ck-synapse-worker-vulkan"),
        "ane-direct-worker" => Some("ck-synapse-worker-ane-direct"),
        _ => None,
    }
}
impl LiveRunner {
    pub fn new(mut options: Options, source: &str) -> Result<Self> {
        options.assets = dunce::canonicalize(&options.assets).map_err(err)?;
        options.checkout = dunce::canonicalize(&options.checkout).map_err(err)?;
        options.mirror = dunce::canonicalize(&options.mirror).map_err(err)?;
        Ok(Self {
            options,
            source: source.into(),
            runtime: tokio::runtime::Runtime::new().map_err(err)?,
            floor: None,
            layout: None,
        })
    }
    fn prepare(&mut self, row: &str) -> Result<()> {
        if let Some((existing, _)) = &self.layout {
            if existing != row {
                return Err(refuse("runner cannot change row during a run"));
            }
            return Ok(());
        }
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(err)?
            .as_nanos();
        let scratch = self
            .options
            .checkout
            .join("crates/synapse-certify-runner/.live")
            .join(format!("run-{}-{unique}", std::process::id()));
        self.layout = Some((
            row.into(),
            crate::assets::Layout::new(&self.options.assets, &scratch, row)?,
        ));
        Ok(())
    }
}
impl Runner for LiveRunner {
    fn probe_floor(&mut self, row: &str, _model: &str) -> Result<String> {
        self.prepare(row)?;
        let layout = &self.layout.as_ref().expect("prepared layout").1;
        let alias = layout
            .worker
            .as_ref()
            .ok_or_else(|| refuse("metal-m5 has no floor probe"))?;
        let output = self.runtime.block_on(async {
            timeout(
                Duration::from_secs(60),
                synapse_core::without_launch_nonce_tokio(Command::new(alias))
                    .arg("--probe-floor")
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .map_err(err)?
            .map_err(err)
        })?;
        let floor: Value = serde_json::from_slice(&output.stdout).map_err(err)?;
        let status = floor["status"].as_str().unwrap_or("refused").to_string();
        self.floor = Some(floor);
        Ok(if output.status.success() {
            status
        } else {
            "refused".into()
        })
    }
    fn observe(&mut self, row: &str, model: &str) -> Result<RunEvidence> {
        self.prepare(row)?;
        self.runtime.block_on(observe(
            &self.options,
            &self.source,
            row,
            model,
            self.floor.as_ref(),
            &self.layout.as_ref().expect("prepared layout").1,
        ))
    }
}

struct DaemonTask(tokio::task::JoinHandle<Result<()>>);
impl Drop for DaemonTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

struct Session {
    child: tokio::process::Child,
    daemon: DaemonTask,
    stream: TcpStream,
    channel: u16,
    epoch: u32,
    corr: u64,
}
impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
        self.daemon.0.abort();
    }
}
async fn rpc(
    stream: &mut TcpStream,
    channel: u16,
    epoch: u32,
    corr: u64,
    body: Value,
) -> Result<Value> {
    let frame = Frame::build(
        FrameType::Request,
        Flags::new(false, Priority::Passive, false),
        channel,
        epoch,
        corr,
        serde_json::to_vec(&body).map_err(err)?,
    )
    .map_err(err)?;
    write_frame(stream, &frame).await.map_err(err)?;
    timeout(Duration::from_secs(3600), async {
        loop {
            let frame = read_frame(stream)
                .await
                .map_err(err)?
                .ok_or_else(|| refuse("candidate connection closed"))?;
            if frame.header.channel == channel
                && frame.header.corr == corr
                && matches!(frame.header.ty, FrameType::Response | FrameType::Error)
            {
                let value: Value = serde_json::from_slice(&frame.body).map_err(err)?;
                if frame.header.ty == FrameType::Error {
                    return Ok(json!({"error": value}));
                }
                return Ok(value);
            }
        }
    })
    .await
    .map_err(err)?
}
impl Session {
    async fn shutdown(&mut self) -> Result<()> {
        self.child.start_kill().map_err(err)?;
        self.child.wait().await.map_err(err)?;
        self.daemon.0.abort();
        Ok(())
    }
    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.corr += 1;
        let value = rpc(
            &mut self.stream,
            self.channel,
            self.epoch,
            self.corr,
            json!({"method": method, "params": params}),
        )
        .await?;
        Ok(value.get("result").cloned().unwrap_or(value))
    }
}
impl crate::catalog::Client for Session {
    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        Session::request(self, method, params).await
    }
}

async fn start(
    options: &Options,
    layout: &crate::assets::Layout,
    endpoint: &str,
) -> Result<Session> {
    let root = &layout.root;
    std::fs::create_dir_all(root.join("data")).map_err(err)?;
    let listener = TcpListener::bind("127.0.0.1:0").await.map_err(err)?;
    let conn = ConnectionInfo {
        schema: SCHEMA_VERSION,
        endpoints: vec![Endpoint {
            host: "127.0.0.1".into(),
            port: listener.local_addr().map_err(err)?.port(),
        }],
        key: generate_key().map_err(err)?,
        daemon_id: generate_daemon_id().map_err(err)?,
        pid: std::process::id(),
        daemon_ver: "synapse-certify".into(),
        wire_version: Some(subc_protocol::PROTOCOL_VERSION),
    };
    let conn_path = root.join("connection.json");
    write_atomic(&conn_path, &conn).map_err(err)?;
    let registry = Arc::new(Registry::default());
    let control = ControlHandler::new(registry).with_storage_config(Some(StorageConfig::Sqlite {
        data_home: root.join("data"),
    }));
    let router = Arc::new(Router::with_control_handler(Arc::new(control)));
    let auth = ServerAuth::new(conn.key.clone(), conn.daemon_id, conn.daemon_ver.clone());
    let daemon = DaemonTask(tokio::spawn(async move {
        serve_listener(listener, router, auth).await.map_err(err)
    }));
    let config = root.join("config.json");
    std::fs::write(
        &config,
        serde_json::to_vec(&crate::assets::config(endpoint)).map_err(err)?,
    )
    .map_err(err)?;
    let mut command = synapse_core::without_launch_nonce_tokio(Command::new(&layout.candidate));
    for (name, value) in layout.environment() {
        if let Some(value) = value {
            command.env(name, value);
        } else {
            command.env_remove(name);
        }
    }
    let child = command
        .current_dir(root)
        .arg("--subc")
        .arg(&conn_path)
        .env("SUBC_MODULE_ID", "synapse")
        .env("SYNAPSE_CONFIG_PATH", &config)
        .env("XDG_DATA_HOME", root.join("data"))
        .env("CORTEXKIT_LEASE_ROOT", root.join("leases"))
        .env("CORTEXKIT_STORE_ROOT", root.join("store"))
        .stderr(std::process::Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(err)?;
    let mut stream = TcpStream::connect(("127.0.0.1", conn.endpoints[0].port))
        .await
        .map_err(err)?;
    authenticate_client(&mut stream, &conn, Duration::from_secs(10))
        .await
        .map_err(err)?;
    let mut session = Session {
        child,
        daemon,
        stream,
        channel: 0,
        epoch: 0,
        corr: 1,
    };
    for attempt in 0..600 {
        session.corr += 1;
        let response = rpc(&mut session.stream, 0, 0, session.corr, json!({"op": "route.open", "target": RouteTarget::ManagementSurface { module_id: "synapse".into() }, "identity": BindIdentity::new(options.checkout.clone(), "certify".to_string(), "certify".to_string())})).await?;
        if let (Some(channel), Some(epoch)) = (
            response["route_channel"].as_u64(),
            response["route_epoch"].as_u64(),
        ) {
            session.channel = channel as u16;
            session.epoch = epoch as u32;
            return Ok(session);
        }
        if session.child.try_wait().map_err(err)?.is_some() {
            return Err(refuse(format!(
                "candidate exited before registration: {response}"
            )));
        }
        if attempt == 599 {
            return Err(refuse(format!(
                "candidate registration timed out: {response}"
            )));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    unreachable!()
}
fn request_params(case: &Value, operation: &str) -> Value {
    if operation == "embed" {
        json!({"text": case["text"], "accept_declared": true})
    } else {
        json!({"query": case["query"], "candidates": [case["document"]], "accept_declared": true})
    }
}
fn observed_output(response: &Value, index: usize, operation: &str) -> serde_json::Result<Output> {
    // ResponseEnvelope flattens the inference payload alongside its metadata.
    if operation == "embed" {
        serde_json::from_value::<Vec<f64>>(response["vectors"][index]["vector"].clone())
            .map(Output::Embedding)
    } else {
        serde_json::from_value::<f64>(response["scores"][index].clone()).map(Output::Score)
    }
}

const INLINE_ITEMS: usize = 64;
const INLINE_TOKENS: usize = 8192;

// Preserve candidate order and test padding with real multi-item calls. Count
// every engine-input token, including query/template and special tokens in each
// pair; their sum bounds the module's inline token limit without understating
// the tokenizer's request size.
fn fixture_requests<'a>(cases: &'a [Value], operation: &str) -> Result<Vec<Vec<&'a Value>>> {
    let mut groups: Vec<Vec<&Value>> = Vec::new();
    let mut grouped = BTreeMap::new();
    for case in cases {
        let count = case["input_ids"]
            .as_array()
            .ok_or_else(|| refuse("fixture input IDs missing"))?
            .len();
        let key = if count == INLINE_TOKENS {
            None
        } else if case["category"] == "batched" {
            Some("batched".to_string())
        } else if operation == "rerank" {
            case["pool"].as_str().map(str::to_string)
        } else {
            None
        };
        if let Some(key) = key {
            let index = *grouped.entry(key).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[index].push(case);
        } else {
            groups.push(vec![case]);
        }
    }
    let mut requests = Vec::new();
    for group in groups {
        let counts = group
            .iter()
            .map(|case| case["input_ids"].as_array().expect("validated IDs").len())
            .collect::<Vec<_>>();
        let ranges = synapse_parity::evaluator::split_pool(&counts, INLINE_ITEMS, INLINE_TOKENS)
            .map_err(err)?;
        for range in ranges {
            requests.push(group[range].to_vec());
        }
    }
    Ok(requests)
}

fn group_params(group: &[&Value], operation: &str) -> Result<(&'static str, Value)> {
    if group.is_empty() {
        return Err(refuse("empty fixture request"));
    }
    if operation == "embed" {
        if group.len() == 1 {
            return Ok(("embed.query", request_params(group[0], operation)));
        }
        let items = group
            .iter()
            .map(|case| json!({"id": case["id"], "text": case["text"]}))
            .collect::<Vec<_>>();
        Ok((
            "embed.batch",
            json!({"items": items, "accept_declared": true}),
        ))
    } else {
        let query = &group[0]["query"];
        if group.iter().any(|case| &case["query"] != query) {
            return Err(refuse("fixture batch queries differ"));
        }
        let candidates = group
            .iter()
            .map(|case| &case["document"])
            .collect::<Vec<_>>();
        Ok((
            "rerank.score",
            json!({"query": query, "candidates": candidates, "accept_declared": true}),
        ))
    }
}

fn sent(snapshot: &Value) -> Result<usize> {
    snapshot["worker_requests"]
        .as_object()
        .ok_or_else(|| refuse("missing worker request counters"))?
        .values()
        .try_fold(0usize, |sum, value| {
            let count = value
                .as_u64()
                .ok_or_else(|| refuse("invalid worker counter"))?;
            sum.checked_add(count as usize)
                .ok_or_else(|| refuse("worker counter overflow"))
        })
}
fn parity_failure_message(
    evaluation: &Evaluation,
    cases: &[Case],
    outputs: &BTreeMap<String, ObservedCase>,
    expected_readout: Option<(u32, u32)>,
) -> String {
    let tolerance_class = match evaluation.tolerance_class {
        synapse_parity::manifest::ToleranceClass::Fp32 => "fp32",
        synapse_parity::manifest::ToleranceClass::Fp16 => "fp16",
    };
    let score_bound = rerank_tolerance(evaluation.tolerance_class);
    let mut lines = vec![
        format!(
            "parity gate failed: model={} profile={}",
            evaluation.model, evaluation.profile_id
        ),
        format!("precision_class={tolerance_class} rerank_score_bound<={score_bound:.6}"),
    ];

    if !evaluation.gates.complete {
        let missing = cases
            .iter()
            .find(|case| !outputs.contains_key(&case.id))
            .map(|case| case.id.as_str())
            .or_else(|| {
                outputs
                    .keys()
                    .find(|id| !cases.iter().any(|case| &case.id == *id))
                    .map(String::as_str)
            })
            .unwrap_or("unavailable");
        lines.push(format!(
            "gate=fixture_completeness bound=completed==expected observed={}/{} worst_case_id={missing}",
            evaluation.completed_fixtures, evaluation.expected_fixtures
        ));
    }

    if !evaluation.gates.score {
        let worst = cases
            .iter()
            .filter_map(|case| {
                let observed = outputs.get(&case.id)?;
                match (&case.output, &observed.output) {
                    (Output::Score(reference), Output::Score(actual)) => Some((
                        case.id.as_str(),
                        if actual.is_finite() {
                            (actual - reference).abs()
                        } else {
                            f64::INFINITY
                        },
                        *actual,
                    )),
                    _ => None,
                }
            })
            .max_by(|left, right| left.1.total_cmp(&right.1));
        if let Some((case_id, error, actual)) = worst {
            let observed = if actual.is_finite() {
                format!("{error:.9}")
            } else {
                "non_finite".into()
            };
            lines.push(format!(
                "gate=score_error bound<={score_bound:.6} observed_worst={observed} worst_case_id={case_id}"
            ));
        } else {
            let case_id = cases
                .iter()
                .find(|case| outputs.contains_key(&case.id))
                .map(|case| case.id.as_str())
                .unwrap_or("unavailable");
            lines.push(format!(
                "gate=score_error bound<={score_bound:.6} observed_worst=type_mismatch worst_case_id={case_id}"
            ));
        }
    }

    if !evaluation.gates.ranking {
        let worst_pool = evaluation
            .metrics
            .pools
            .iter()
            .filter(|(_, ranking)| !ranking.passed)
            .min_by(|left, right| left.1.tau.total_cmp(&right.1.tau));
        if let Some((pool, ranking)) = worst_pool {
            let mut near_ties = 0;
            let mut worst_pair: Option<(&str, &str, f64, f64, f64)> = None;
            let pool_cases = cases
                .iter()
                .filter(|case| case.pool.as_deref() == Some(pool.as_str()))
                .filter_map(|case| {
                    let Output::Score(reference) = &case.output else {
                        return None;
                    };
                    let Output::Score(actual) = &outputs.get(&case.id)?.output else {
                        return None;
                    };
                    actual
                        .is_finite()
                        .then_some((case.id.as_str(), *reference, *actual))
                })
                .collect::<Vec<_>>();
            for left in 0..pool_cases.len() {
                for right in left + 1..pool_cases.len() {
                    let (left_id, left_reference, left_actual) = pool_cases[left];
                    let (right_id, right_reference, right_actual) = pool_cases[right];
                    let gap = (left_reference - right_reference).abs();
                    if gap < 0.01 {
                        near_ties += 1;
                        continue;
                    }
                    let (high_id, high_actual, low_id, low_actual) =
                        if left_reference > right_reference {
                            (left_id, left_actual, right_id, right_actual)
                        } else {
                            (right_id, right_actual, left_id, left_actual)
                        };
                    if high_actual <= low_actual
                        && worst_pair.is_none_or(|(_, _, worst_gap, _, _)| gap > worst_gap)
                    {
                        worst_pair = Some((high_id, low_id, gap, high_actual, low_actual));
                    }
                }
            }
            let pair = worst_pair.map_or_else(
                || "unavailable".to_string(),
                |(high, low, gap, high_score, low_score)| {
                    format!(
                        "{high}/{low} reference_gap={gap:.9} observed_scores={high_score:.9}/{low_score:.9}"
                    )
                },
            );
            lines.push(format!(
                "gate=ranking_order_or_tau bound=tau>=0.99,top10=true observed_tau={:.9} top10={} pool={pool} worst_case_pair={pair} near_ties_excluded={near_ties}(reference_gap<0.01)",
                ranking.tau, ranking.top10
            ));
        } else {
            lines.push(
                "gate=ranking_order_or_tau bound=tau>=0.99,top10=true observed=pool_metrics_unavailable worst_case_pair=unavailable near_ties_excluded=unavailable(reference_gap<0.01)".into(),
            );
        }
    }

    if !evaluation.gates.semantics {
        let reason = evaluation
            .failures
            .iter()
            .find(|failure| failure.starts_with("qwen_"))
            .map(String::as_str)
            .unwrap_or("token_readout_mismatch");
        let mismatch = cases.iter().find_map(|case| {
            let observed = outputs.get(&case.id)?;
            let token_mismatch = observed.input_ids != case.input_ids;
            let readout_mismatch =
                expected_readout.is_some_and(|expected| observed.readout != Some(expected));
            (token_mismatch || readout_mismatch).then_some((case, observed))
        });
        let details = mismatch.map_or_else(
            || "worst_case_id=unavailable".to_string(),
            |(case, observed)| {
                format!(
                    "worst_case_id={} expected_readout={expected_readout:?} observed_readout={:?} input_ids_match={}",
                    case.id,
                    observed.readout,
                    observed.input_ids == case.input_ids
                )
            },
        );
        lines.push(format!(
            "gate=token_readout_or_template failed={reason} {details}"
        ));
    }

    if !evaluation.gates.coverage_8192 {
        let case = cases.iter().find(|case| case.input_ids.len() == 8192);
        let (case_id, observed_tokens) = case.map_or(("unavailable", 0), |case| {
            (
                case.id.as_str(),
                outputs
                    .get(&case.id)
                    .map_or(0, |observed| observed.input_ids.len()),
            )
        });
        lines.push(format!(
            "gate=token_readout_8192 bound=exact_8192_tokens observed_tokens={observed_tokens} worst_case_id={case_id}"
        ));
    }

    if !evaluation.gates.coverage {
        lines.push(format!(
            "gate=fixture_category_coverage bound=all_required_categories observed={:?}",
            evaluation.failures
        ));
    }

    for (gate, passed) in [
        ("dimension", evaluation.gates.dimension),
        ("finite", evaluation.gates.finite),
        ("unit_norm", evaluation.gates.unit_norm),
        ("cosine", evaluation.gates.cosine),
    ] {
        if !passed {
            lines.push(format!(
                "gate={gate} failed; inspect fixture outputs for the observed value and case id"
            ));
        }
    }

    lines.join("\n")
}

async fn observe(
    options: &Options,
    source: &str,
    row: &str,
    model: &str,
    floor: Option<&Value>,
    layout: &crate::assets::Layout,
) -> Result<RunEvidence> {
    let parity_root = options.checkout.join("bench/parity");
    let manifest = synapse_certify::self_check::manifest()?;
    let index: Value =
        serde_json::from_slice(include_bytes!("../../../bench/parity/fixtures/index.json"))
            .map_err(err)?;
    let fixture_id = synapse_parity::evaluator::fixture_set_id(&manifest, model);
    let fixture_bytes = std::fs::read(
        parity_root.join(
            index[&fixture_id]["path"]
                .as_str()
                .ok_or_else(|| refuse("fixture path missing"))?,
        ),
    )
    .map_err(err)?;
    let fixtures = FixtureSet::from_bytes(
        &manifest,
        model,
        &fixture_bytes,
        index[&fixture_id]["sha256"]
            .as_str()
            .ok_or_else(|| refuse("fixture digest missing"))?,
    )
    .map_err(err)?;
    let document: Value = serde_json::from_slice(&fixture_bytes).map_err(err)?;
    let cases = document["cases"]
        .as_array()
        .ok_or_else(|| refuse("fixture cases missing"))?;
    let mirror = crate::mirror::Mirror::start(&options.mirror).await?;
    let mut candidate = start(options, layout, &mirror.endpoint).await?;
    let deadline = tokio::time::Instant::now() + crate::catalog::RUN_BUDGET;
    let pin = crate::catalog::install(&mut candidate, row, model, deadline).await?;
    let operation = pin.task.clone();
    let operation = operation.as_str();
    let method = if operation == "embed" {
        "embed.query"
    } else {
        "rerank.score"
    };
    let profile_id = format!("{model}.{}", lane(row));
    let mut session = crate::catalog::CertifiedSession::new(&mut candidate, pin, deadline);
    let mut outputs = BTreeMap::new();
    let mut long_response = None;
    let mut long_request_count = 0;
    for group in fixture_requests(cases, operation)? {
        let case = group[0];
        let is_long = case["input_ids"]
            .as_array()
            .is_some_and(|ids| ids.len() == 8192);
        let before_long = if is_long {
            Some(sent(&session.observations().await?)?)
        } else {
            None
        };
        let (request_method, params) = group_params(&group, operation)?;
        let response = session.fixture(request_method, params).await?;
        let ids: Vec<Vec<u32>> =
            serde_json::from_value(response["observation"]["input_ids"].clone()).map_err(err)?;
        if ids.len() != group.len() {
            return Err(refuse("fixture observation count mismatch"));
        }
        let readout = response["observation"]["readout_ids"]
            .as_array()
            .and_then(|ids| Some((ids.first()?.as_u64()? as u32, ids.get(1)?.as_u64()? as u32)));
        if is_long && ids[0].len() == 8192 {
            long_response = Some(response.clone());
            long_request_count = sent(&session.observations().await?)?
                .checked_sub(before_long.expect("long snapshot captured"))
                .ok_or_else(|| refuse("worker counter regressed on 8192 input"))?;
        }
        for (index, (case, input_ids)) in group.into_iter().zip(ids).enumerate() {
            let output = observed_output(&response, index, operation).map_err(err)?;
            outputs.insert(
                case["id"]
                    .as_str()
                    .ok_or_else(|| refuse("fixture id missing"))?
                    .to_string(),
                ObservedCase {
                    output,
                    input_ids,
                    readout,
                },
            );
        }
    }
    let fingerprint = session.pin.fingerprint.clone();
    let evaluation =
        evaluate(&manifest, &profile_id, &fingerprint, &fixtures, &outputs).map_err(err)?;
    let parity: Parity =
        serde_json::from_value(serde_json::to_value(&evaluation).map_err(err)?).map_err(err)?;
    let expected_readout = manifest.models.get(model).and_then(|model| {
        model
            .grammar
            .readout
            .yes
            .as_ref()
            .zip(model.grammar.readout.no.as_ref())
            .map(|(yes, no)| (yes.id, no.id))
    });
    let parity_failure = (!parity.passed())
        .then(|| parity_failure_message(&evaluation, fixtures.cases(), &outputs, expected_readout));
    let long_case = cases
        .iter()
        .find(|case| {
            case["input_ids"]
                .as_array()
                .is_some_and(|ids| ids.len() == 8192)
        })
        .ok_or_else(|| refuse("8192 fixture missing"))?;
    let mut oversized = long_case.clone();
    let field = if operation == "embed" {
        "text"
    } else {
        "document"
    };
    oversized[field] = json!(format!(
        "{} a",
        long_case[field]
            .as_str()
            .ok_or_else(|| refuse("long fixture text missing"))?
    ));
    let before = session.observations().await?;
    let rejected = session
        .oversized(method, request_params(&oversized, operation))
        .await?;
    let after = session.observations().await?;
    if rejected["error"]["code"] == "sequence_too_long"
        && rejected["error"]["details"]["tokens"] != 8193
    {
        return Err(refuse(format!(
            "8193 admission input composed to a different length: {rejected}"
        )));
    }
    let delta = sent(&after)?
        .checked_sub(sent(&before)?)
        .ok_or_else(|| refuse("worker counter regressed"))?;
    let response = long_response.unwrap_or(Value::Null);
    let admission = Admission {
        tokens_8192: Outcome {
            outcome: if response.is_null() {
                "not_processed"
            } else {
                "processed"
            }
            .into(),
            truncated: any_truncated(&response),
            diverted: !response["job_id"].is_null(),
            worker_requests: long_request_count,
        },
        tokens_8193: Outcome {
            outcome: rejected["error"]["code"]
                .as_str()
                .unwrap_or("missing_error")
                .into(),
            truncated: false,
            diverted: !rejected["job_id"].is_null(),
            worker_requests: delta,
        },
    };
    let final_snapshot = session.observations().await?;
    let inventories: Vec<Inventory> =
        serde_json::from_value(final_snapshot["inventories"].clone()).map_err(err)?;
    let artifacts = layout.artifacts.clone();
    let machine = machine(row, floor)?;
    candidate.shutdown().await?;
    if let Some(message) = parity_failure {
        return Err(refuse(message));
    }
    Ok(RunEvidence {
        source_commit: source.into(),
        machine,
        artifacts,
        operation: operation.into(),
        profile_id,
        fingerprint,
        fixture_set_id: fixture_id,
        expected_fixtures: fixtures.cases().len(),
        tolerance_class: parity.tolerance_class.clone(),
        parity: (!outputs.is_empty()).then_some(parity),
        admission,
        layer_count: manifest
            .model(model)
            .map_err(err)?
            .architecture
            .int("num_hidden_layers")
            .map_err(err)? as usize,
        inventories,
        admitted_count: final_snapshot["admitted_count"]
            .as_u64()
            .ok_or_else(|| refuse("independent ADMITTED counter missing"))?
            as usize,
    })
}
// `Session::request` returns the unwrapped `result`, so an embed or rerank
// reply carries `vectors` or `scores` and `truncation_disclosures` at its top
// level; there is no `payload` envelope.

fn any_truncated(response: &Value) -> bool {
    response["truncation_disclosures"]
        .as_array()
        .is_some_and(|items| items.iter().any(|item| item["truncated"] == true))
}

fn machine(row: &str, floor: Option<&Value>) -> Result<Value> {
    #[cfg(target_os = "macos")]
    {
        let output =
            synapse_core::without_launch_nonce(std::process::Command::new("system_profiler"))
                .args(["SPHardwareDataType", "-json"])
                .output()
                .map_err(err)?;
        let hardware: Value = serde_json::from_slice(&output.stdout).map_err(err)?;
        let hardware = &hardware["SPHardwareDataType"][0];
        if !matches!(row, "metal-m5" | "ane-m5")
            || !hardware["chip_type"]
                .as_str()
                .is_some_and(|chip| chip.contains("M5"))
        {
            return Err(refuse(
                "machine does not match the requested M5 certification row",
            ));
        }
        let (Some(_), Some(platform_uuid)) = (
            hardware["machine_model"].as_str(),
            hardware["platform_UUID"].as_str(),
        ) else {
            return Err(refuse("Apple machine identifiers missing"));
        };
        let os = synapse_core::without_launch_nonce(std::process::Command::new("sw_vers"))
            .arg("-productVersion")
            .output()
            .map_err(err)?;
        Ok(
            json!({"model_identifier": hardware["machine_model"], "platform_uuid_sha256": synapse_certify::platform_uuid_sha256(platform_uuid), "os": String::from_utf8_lossy(&os.stdout).trim(), "driver": if row == "metal-m5" { json!(format!("Metal on macOS {}", String::from_utf8_lossy(&os.stdout).trim())) } else { floor.cloned().unwrap_or(Value::Null) }}),
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        machine_non_apple(row, floor)
    }
}

#[cfg(not(target_os = "macos"))]
fn machine_non_apple(row: &str, floor: Option<&Value>) -> Result<Value> {
    fn command(program: &str, args: &[&str]) -> Result<String> {
        let output = synapse_core::without_launch_nonce(std::process::Command::new(program))
            .args(args)
            .output()
            .map_err(err)?;
        if !output.status.success() {
            return Err(refuse(format!(
                "machine probe {program}: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
    #[cfg(target_os = "windows")]
    let (product, os) = (
        command("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).Model"] )?,
        command("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).Version"] )?,
    );
    #[cfg(not(target_os = "windows"))]
    let (product, os) = (
        std::fs::read_to_string("/sys/class/dmi/id/product_name")
            .map_err(err)?
            .trim()
            .to_string(),
        command("uname", &["-srv"])?,
    );
    let (gpu, driver) = if row.starts_with("cuda-") {
        let output = command(
            "nvidia-smi",
            &[
                "--query-gpu=pci.device_id,name,uuid,driver_version",
                "--format=csv,noheader",
            ],
        )?;
        let lines = output.lines().collect::<Vec<_>>();
        if lines.len() != 1 {
            return Err(refuse(
                "machine identity requires exactly one visible NVIDIA GPU",
            ));
        }
        let fields = lines[0].split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(refuse("invalid nvidia-smi machine identity"));
        }
        let pci = u32::from_str_radix(fields[0].trim_start_matches("0x"), 16).map_err(err)?;
        (
            json!({"vendor_id": pci & 0xffff, "device_id": pci >> 16, "name": fields[1], "uuid_sha256": synapse_certify::gpu_uuid_sha256(fields[2])}),
            fields[3].to_string(),
        )
    } else {
        let output = command("vulkaninfo", &["--summary"])?;
        let index = floor
            .and_then(|f| f["observed"]["index"].as_u64())
            .ok_or_else(|| refuse("floor probe omitted Vulkan adapter index"))?;
        let section = output
            .split(&format!("GPU{index}:"))
            .nth(1)
            .ok_or_else(|| refuse("vulkaninfo omitted selected adapter"))?;
        let mut properties = BTreeMap::new();
        for line in section
            .lines()
            .take_while(|line| !line.trim_start().starts_with("GPU"))
        {
            if let Some((key, value)) = line.split_once('=') {
                properties.insert(key.trim(), value.trim());
            }
        }
        let get = |key| {
            properties
                .get(key)
                .copied()
                .ok_or_else(|| refuse(format!("Vulkan machine identifier missing: {key}")))
        };
        (
            json!({"vendor_id": get("vendorID")?, "device_id": get("deviceID")?, "name": get("deviceName")?, "uuid_sha256": synapse_certify::gpu_uuid_sha256(get("deviceUUID")?)}),
            format!("{} {}", get("driverName")?, get("driverInfo")?),
        )
    };
    let mut machine =
        json!({"system_product_name": product, "gpu": gpu, "os": os, "driver": driver});
    if row.starts_with("cuda-") {
        machine["driver_api"] = floor.ok_or_else(|| refuse("CUDA floor observation missing"))?
            ["observed"]["driver_api"]
            .clone();
    }
    if let Ok(instance) = std::env::var("SYNAPSE_CERTIFY_INSTANCE_ID") {
        machine["instance_id"] = json!(instance);
    }
    Ok(machine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // Run a short catalog download/load/serve probe against a signed release
    // module. The runner gives it a ckdev alias and an isolated store/lease root.
    // Run: CKDEV_RERANK_ASSETS=<release-assets> CKDEV_RERANK_MIRROR=<pinned-mirror>
    // CKDEV_RERANK_DOMAIN=probabilities cargo test --release --locked
    // -p synapse-certify-runner catalog_metal_rerank_fixture_readout_probe
    // -- --ignored --nocapture (use domain=logits to reproduce the old readout).
    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[ignore = "requires signed release module assets and the pinned local mirror"]
    async fn catalog_metal_rerank_fixture_readout_probe() {
        let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let options = Options {
            assets: std::env::var_os("CKDEV_RERANK_ASSETS")
                .expect("CKDEV_RERANK_ASSETS")
                .into(),
            mirror: std::env::var_os("CKDEV_RERANK_MIRROR")
                .expect("CKDEV_RERANK_MIRROR")
                .into(),
            checkout: checkout.clone(),
        };
        let domain = std::env::var("CKDEV_RERANK_DOMAIN").expect("logits or probabilities");
        assert!(matches!(domain.as_str(), "logits" | "probabilities"));
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!(".live/readout-probe-{}", std::process::id()));
        let layout = crate::assets::Layout::new(&options.assets, &root, "metal-m5").unwrap();
        let mirror = crate::mirror::Mirror::start(&options.mirror).await.unwrap();
        let mut candidate = start(&options, &layout, &mirror.endpoint).await.unwrap();
        let deadline = tokio::time::Instant::now() + crate::catalog::RUN_BUDGET;
        let pin = crate::catalog::install(
            &mut candidate,
            "metal-m5",
            "gte-reranker-modernbert-base",
            deadline,
        )
        .await
        .unwrap();
        let fingerprint = pin.fingerprint.clone();
        let mut session = crate::catalog::CertifiedSession::new(&mut candidate, pin, deadline);
        let bytes = std::fs::read(checkout.join("bench/parity/fixtures/gte-reranker-modernbert-base/gte-reranker-modernbert-base.ref-v1.transformers-5.16.1.seed-0.json")).unwrap();
        let document: Value = serde_json::from_slice(&bytes).unwrap();
        let cases = document["cases"].as_array().unwrap();
        // Keep the original certification request shape: the first 64-item pool.
        let groups = fixture_requests(cases, "rerank").unwrap();
        let group = groups
            .iter()
            .find(|group| group[0]["id"] == "pool100-000")
            .unwrap();
        let (method, params) = group_params(group, "rerank").unwrap();
        assert_eq!(method, "rerank.score");
        let response = session.fixture(method, params).await.unwrap();
        let mut report = Vec::new();
        for index in [0, 7, 42, 63] {
            let case = group[index];
            assert_eq!(
                response["observation"]["input_ids"][index], case["input_ids"],
                "{}",
                case["id"]
            );
            let observed = response["scores"][index].as_f64().unwrap();
            let reference = case["output"].as_f64().unwrap();
            let sigmoid = 1.0 / (1.0 + (-observed).exp());
            let comparable = if domain == "logits" {
                sigmoid
            } else {
                observed
            };
            assert!(
                (comparable - reference).abs() < 0.001,
                "{}: {observed} versus {reference}",
                case["id"]
            );
            if domain == "logits" {
                assert!((observed - reference).abs() > 0.1);
            }
            report.push(json!({"id":case["id"], "reference":reference, "observed":observed, "sigmoid_observed":sigmoid, "abs_error":(observed-reference).abs(), "sigmoid_abs_error":(sigmoid-reference).abs(), "input_ids_match":true}));
        }
        candidate.shutdown().await.unwrap();
        let report = json!({"domain":domain, "fingerprint":fingerprint, "cases":report});
        println!("{report}");
        std::fs::write(
            root.join("readout.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn floor_probe_and_worker_environment_use_one_alias_per_run() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!(".live/floor-test-{}", std::process::id()));
        let assets = root.join("assets/linux-x64");
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("ck-synapse"), b"candidate").unwrap();
        let worker = assets.join("ck-synapse-worker-cuda");
        std::fs::write(&worker, "#!/bin/sh\n[ \"$1\" = '--probe-floor' ] || exit 2\nprintf '{\"status\":\"ok\",\"executable\":\"%s\"}\\n' \"$0\"\n").unwrap();
        std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut runner = LiveRunner::new(
            Options {
                assets: root.join("assets"),
                checkout: root.clone(),
                mirror: root.clone(),
            },
            &"a".repeat(40),
        )
        .unwrap();
        assert_eq!(
            runner
                .probe_floor("cuda-linux-nvidia", "gte-modernbert-base")
                .unwrap(),
            "ok"
        );
        let alias = runner
            .layout
            .as_ref()
            .unwrap()
            .1
            .worker
            .as_ref()
            .unwrap()
            .clone();
        assert_eq!(
            runner.floor.as_ref().unwrap()["executable"],
            alias.to_str().unwrap()
        );
        runner.prepare("cuda-linux-nvidia").unwrap();
        let layout = &runner.layout.as_ref().unwrap().1;
        assert_eq!(layout.worker.as_ref().unwrap(), &alias);
        assert_eq!(
            layout
                .environment()
                .into_iter()
                .find(|(name, _)| name == "SYNAPSE_OWNED_CUDA_WORKER_BIN")
                .unwrap()
                .1,
            Some(alias.clone())
        );
        assert_eq!(
            std::fs::metadata(&worker).unwrap().ino(),
            std::fs::metadata(&alias).unwrap().ino()
        );
        assert_ne!(layout.candidate.parent(), alias.parent());
        assert_eq!(
            std::fs::read_dir(alias.parent().unwrap()).unwrap().count(),
            1
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    // Shapes copied from real module replies captured during an ane-m5 run.
    #[test]
    fn module_replies_are_read_at_their_top_level() {
        let embed = json!({"dims": 2, "vectors": [[0.6, 0.8]],
            "truncation_disclosures": [{"effective_tokens": 512, "submitted_tokens": 512, "truncated": false}]});
        let rerank = json!({"dims": 1, "scores": [0.99],
            "truncation_disclosures": [{"truncated": false}, {"truncated": false}]});
        assert!(!any_truncated(&embed));
        let mut cut = rerank.clone();
        cut["truncation_disclosures"][1]["truncated"] = json!(true);
        assert!(any_truncated(&cut));
    }

    #[test]
    fn development_aliases_preserve_attested_binary_identity() {
        use sha2::{Digest, Sha256};
        let root =
            std::env::temp_dir().join(format!("synapse-certify-alias-{}", std::process::id()));
        let assets = root.join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        for role in [
            "ck-synapse",
            "ck-synapse-worker-cuda",
            "ck-synapse-worker-vulkan",
            "ck-synapse-worker-ane-direct",
        ] {
            let built = assets.join(crate::assets::binary_file(
                if cfg!(windows) {
                    "cuda-windows-nvidia"
                } else {
                    "metal-m5"
                },
                role,
            ));
            std::fs::write(&built, role.as_bytes()).unwrap();
            let alias =
                synapse_core::dev_binary::ckdev_binary_hard_link(&built, root.join("run")).unwrap();
            assert_eq!(
                alias.file_name().unwrap().to_string_lossy(),
                format!(
                    "ckdev-{}{}",
                    role.strip_prefix("ck-").unwrap(),
                    std::env::consts::EXE_SUFFIX
                )
            );
            assert_eq!(
                Sha256::digest(std::fs::read(&built).unwrap()),
                Sha256::digest(std::fs::read(&alias).unwrap())
            );
            assert!(
                built.is_file(),
                "the attested production file name must remain available"
            );
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                assert_eq!(
                    std::fs::metadata(&built).unwrap().ino(),
                    std::fs::metadata(alias).unwrap().ino()
                );
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn decodes_flattened_wire_vectors_and_scores_in_item_order() {
        let embed = json!({"fingerprint": "identity", "vectors": [{"id":"a", "vector":[1.0, 2.0]}, {"id":"b", "vector":[3.0, 4.0]}]});
        let rerank = json!({"fingerprint": "identity", "scores": [0.25, 0.75]});
        assert_eq!(
            observed_output(&embed, 0, "embed").unwrap(),
            Output::Embedding(vec![1.0, 2.0])
        );
        assert_eq!(
            observed_output(&embed, 1, "embed").unwrap(),
            Output::Embedding(vec![3.0, 4.0])
        );
        assert_eq!(
            observed_output(&rerank, 0, "rerank").unwrap(),
            Output::Score(0.25)
        );
        assert_eq!(
            observed_output(&rerank, 1, "rerank").unwrap(),
            Output::Score(0.75)
        );
        assert!(observed_output(&embed, 2, "embed").is_err());
    }

    #[test]
    fn parity_failure_reports_score_bound_class_and_worst_fixture() {
        use synapse_parity::manifest::{Operation, ToleranceClass};

        let case = Case {
            id: "rerank-worst".into(),
            category: "short".into(),
            input_ids: vec![1, 2],
            output: Output::Score(0.25),
            pool: None,
        };
        let outputs = BTreeMap::from([(
            case.id.clone(),
            ObservedCase {
                output: Output::Score(0.27),
                input_ids: case.input_ids.clone(),
                readout: None,
            },
        )]);
        let evaluation = Evaluation {
            model: "gte-reranker-modernbert-base".into(),
            operation: Operation::Rerank,
            profile_id: "gte-reranker-modernbert-base.owned-metal".into(),
            fingerprint: "catalog-pin".into(),
            fixture_set_id: "fixture-set".into(),
            tolerance_class: ToleranceClass::Fp32,
            completed_fixtures: 1,
            expected_fixtures: 1,
            metrics: synapse_parity::evaluator::Metrics {
                max_absolute_error: Some(0.02),
                ..Default::default()
            },
            gates: synapse_parity::evaluator::Gates {
                complete: true,
                dimension: true,
                finite: true,
                unit_norm: true,
                cosine: true,
                score: false,
                ranking: true,
                coverage_8192: true,
                coverage: true,
                semantics: true,
            },
            failures: vec![],
        };

        let message = parity_failure_message(&evaluation, &[case], &outputs, None);
        assert!(message.contains("precision_class=fp32 rerank_score_bound<=0.005000"));
        assert!(message.contains(
            "gate=score_error bound<=0.005000 observed_worst=0.020000000 worst_case_id=rerank-worst"
        ));
        assert!(message.lines().count() <= 20);
    }

    #[test]
    fn committed_batches_and_pools_use_bounded_multi_item_requests() {
        let parity = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity");
        let index: Value =
            serde_json::from_slice(&std::fs::read(parity.join("fixtures/index.json")).unwrap())
                .unwrap();
        for entry in index.as_object().unwrap().values() {
            let bytes = std::fs::read(parity.join(entry["path"].as_str().unwrap())).unwrap();
            assert_eq!(
                synapse_parity::canonical::sha256_hex(&bytes),
                entry["sha256"].as_str().unwrap()
            );
            let document: Value = serde_json::from_slice(&bytes).unwrap();
            let operation = document["operation"].as_str().unwrap();
            let cases = document["cases"].as_array().unwrap();
            let requests = fixture_requests(cases, operation).unwrap();
            assert_eq!(
                requests
                    .iter()
                    .flatten()
                    .map(|case| &case["id"])
                    .collect::<Vec<_>>(),
                cases.iter().map(|case| &case["id"]).collect::<Vec<_>>()
            );
            for request in &requests {
                assert!(request.len() <= 64);
                assert!(
                    request
                        .iter()
                        .map(|case| case["input_ids"].as_array().unwrap().len())
                        .sum::<usize>()
                        <= 8192
                );
                if request[0]["input_ids"].as_array().unwrap().len() == 8192 {
                    assert_eq!(request.len(), 1);
                }
            }
            let batched = requests
                .iter()
                .find(|request| request[0]["category"] == "batched")
                .unwrap();
            let (method, params) = group_params(batched, operation).unwrap();
            if operation == "embed" {
                assert_eq!(method, "embed.batch");
                assert_eq!(params["items"].as_array().unwrap().len(), 6);
                assert_eq!(params["items"][0]["id"], "batch-0");
            } else {
                assert_eq!(method, "rerank.score");
                assert_eq!(params["candidates"].as_array().unwrap().len(), 4);
                let pool = requests
                    .iter()
                    .filter(|request| request[0]["pool"] == "pool-100")
                    .collect::<Vec<_>>();
                assert_eq!(
                    pool.iter().map(|request| request.len()).collect::<Vec<_>>(),
                    [64, 36]
                );
                assert_eq!(pool[0][0]["id"], "pool100-000");
                assert_eq!(pool[1][0]["id"], "pool100-064");
            }
        }
    }
}
