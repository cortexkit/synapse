//! Module-side cost of one AFT-shaped `embed.batch` call (64 rows, 1,024-dim
//! vectors, about 120 tokens per row) with an engine that does almost no work.
//!
//! The deterministic test engine replaces the Neural Engine, so whatever time
//! remains is module, transport and client codec cost. The profiling run is
//! an ignored test because its numbers only mean something in a release build
//! on a quiet machine; see
//! docs/evidence/embed-batch-module-cost/README.md for the method.

use super::*;
use std::collections::BTreeMap;

const ROWS: usize = 64;
const DIMS: usize = 1_024;

/// Code-like text: a row is a short Rust function, which the Qwen3 tokenizer
/// splits into roughly 120 tokens. The row index keeps every row distinct.
fn code_row(index: usize) -> String {
    format!(
        "fn handle_request_{index}(state: &ModuleState, params: Value) -> Result<Reply, Error> {{\n    \
         let items = parse_items(&params)?;\n    \
         for (position, item) in items.iter().enumerate() {{\n        \
         if item.text.len() > MAX_TEXT_BYTES_{index} {{\n            \
         return Err(Error::too_long(position, item.text.len()));\n        }}\n    }}\n    \
         let reply = Reply::new(items.len(), state.generation);\n    \
         tracing::debug!(count = items.len(), \"request {index} parsed\");\n    \
         Ok(reply)\n}}"
    )
}

/// Whitespace text for the fixture word-level tokenizer: 120 known words.
fn word_row(index: usize) -> String {
    (0..120)
        .map(|word| ((index + word) % 64).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The Qwen3-Embedding-0.6B `tokenizer.json`: SYNAPSE_MODULE_COST_TOKENIZER if
/// set, else the first local Hugging Face cache snapshot. Without one, the
/// real-tokenizer timing is skipped and reported as null.
fn qwen_tokenizer() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_MODULE_COST_TOKENIZER") {
        return Some(PathBuf::from(path));
    }
    let snapshots = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("tokenizer.json"))
        .find(|path| path.exists())
}

/// Preload one deterministic 1,024-dim embedding lane with the fixture
/// word-level tokenizer. The deterministic lane must pass its probe before it
/// serves, and the probe's reference vectors assume the fixture tokenizer, so
/// the real Qwen3 tokenizer cannot be swapped in here; its cost is measured
/// in-process instead (see `qwen_tokenize_ms`).
fn module_cost_preload() -> String {
    let root = deterministic_source_dir("synapse-module-cost");
    serde_json::json!([{
        "model_id": "test-minilm", "engine": "test-deterministic", "task": "embed",
        "model_path": root.join("model.safetensors"), "tokenizer_path": root.join("tokenizer.json"),
        "format": "test-deterministic", "pooling": "mean", "normalize": true,
        "max_tokens": 512, "quant": "fp32"
    }])
    .to_string()
}

fn rows() -> Vec<Value> {
    (0..ROWS)
        .map(|index| {
            serde_json::json!({
                "id": format!("chunk-{index:03}"),
                "text": word_row(index),
            })
        })
        .collect()
}

/// Time the module's own tokenizer wrapper on 64 code rows with the real
/// Qwen3-Embedding tokenizer: the same call `embed.batch` makes, without the
/// module around it. Returns per-run milliseconds and the total token count.
fn qwen_tokenize_ms(path: &Path, runs: usize) -> (Vec<f64>, u64) {
    let tokenizer = synapse_core::SanitizedTokenizer::from_file(
        path,
        synapse_core::TokenizerConfig { max_tokens: 8_192 },
    )
    .expect("load Qwen3 tokenizer");
    let texts = (0..ROWS).map(code_row).collect::<Vec<_>>();
    let mut tokens = 0;
    let mut samples = Vec::new();
    for run in 0..runs + 2 {
        let started = Instant::now();
        let batch = tokenizer
            .tokenize_batch(texts.iter().map(String::as_str))
            .expect("tokenize code rows");
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        tokens = batch
            .real_token_counts
            .iter()
            .map(|&count| u64::from(count))
            .sum();
        // The first two runs are warm-up, matching the module calls.
        if run >= 2 {
            samples.push(elapsed);
        }
    }
    (samples, tokens)
}

/// One client-observed call: request encode, send-to-reply-frame, JSON parse
/// and vector extraction, plus the module stage records it produced.
struct CallSample {
    encode_ms: f64,
    roundtrip_ms: f64,
    parse_ms: f64,
    extract_ms: f64,
    reply_bytes: usize,
    tokens: u64,
    stages: BTreeMap<String, (f64, usize)>,
}

fn extract_vectors(result: &Value) -> Vec<Vec<f32>> {
    result["vectors"]
        .as_array()
        .expect("vectors array")
        .iter()
        .map(|row| {
            row["vector"]
                .as_array()
                .expect("JSON vector array")
                .iter()
                .map(|value| value.as_f64().expect("finite vector value") as f32)
                .collect()
        })
        .collect()
}

fn read_new_stage_records(
    profile_dir: &Path,
    offsets: &mut BTreeMap<PathBuf, usize>,
) -> Vec<Value> {
    let mut records = Vec::new();
    for entry in std::fs::read_dir(profile_dir)
        .unwrap()
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("module-") && name.ends_with(".jsonl"))
        {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let offset = offsets.entry(path).or_default();
        for line in text[*offset..].lines().filter(|line| !line.is_empty()) {
            records.push(serde_json::from_str(line).unwrap());
        }
        *offset = text.len();
    }
    records
}

async fn one_call(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    corr: u64,
    body: &Value,
) -> (CallSample, Value) {
    let encode_started = Instant::now();
    let request = serde_json::to_vec(body).unwrap();
    let encode_ms = encode_started.elapsed().as_secs_f64() * 1_000.0;
    let roundtrip_started = Instant::now();
    let frame = Frame::build(
        FrameType::Request,
        Flags::new(false, Priority::Interactive, false),
        route.channel,
        route.epoch,
        corr,
        request,
    )
    .unwrap();
    write_frame(consumer, &frame).await.unwrap();
    let reply = loop {
        let frame = read_frame_timeout(consumer).await;
        if frame.header.corr == corr {
            break frame;
        }
    };
    let roundtrip_ms = roundtrip_started.elapsed().as_secs_f64() * 1_000.0;
    assert_eq!(reply.header.ty, FrameType::Response, "reply frame type");
    let parse_started = Instant::now();
    let value: Value = serde_json::from_slice(&reply.body).unwrap();
    let parse_ms = parse_started.elapsed().as_secs_f64() * 1_000.0;
    let result = &value["result"];
    assert!(result["error"].is_null(), "embed.batch failed: {value}");
    assert!(
        result["job_id"].is_null(),
        "the call must stay inline: {result}"
    );
    let extract_started = Instant::now();
    let vectors = extract_vectors(result);
    let extract_ms = extract_started.elapsed().as_secs_f64() * 1_000.0;
    assert_eq!(vectors.len(), ROWS);
    assert!(vectors.iter().all(|vector| vector.len() == DIMS));
    let tokens = result["real_token_counts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|count| count.as_u64().unwrap())
        .sum();
    (
        CallSample {
            encode_ms,
            roundtrip_ms,
            parse_ms,
            extract_ms,
            reply_bytes: reply.body.len(),
            tokens,
            stages: BTreeMap::new(),
        },
        value,
    )
}

fn summarize(label: &str, mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let median = if values.len() % 2 == 1 {
        values[values.len() / 2]
    } else {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    };
    eprintln!(
        "module-cost {label:<28} median={median:9.3} min={:9.3} max={:9.3}",
        values[0],
        values[values.len() - 1]
    );
    serde_json::json!({"median_ms": median, "min_ms": values[0], "max_ms": values[values.len() - 1]})
}

// Ignored rather than skipped at run time: CI fails the e2e lane on any
// "skipping" line, and the numbers only mean something in a release build on a
// quiet machine. Run it with `--ignored` as the evidence README describes.
#[tokio::test]
#[ignore = "profiling run: release build on a quiet machine, see docs/evidence/embed-batch-module-cost"]
async fn embed_batch_module_cost_profile() {
    let runs = std::env::var("SYNAPSE_MODULE_COST_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(9)
        .max(5);
    let out_dir = std::env::var("SYNAPSE_MODULE_COST_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| unique_temp_dir("synapse-module-cost-out"));
    let profile_dir = unique_temp_dir("synapse-module-cost-profile");
    std::fs::create_dir_all(&profile_dir).unwrap();
    std::fs::create_dir_all(&out_dir).unwrap();

    let preload = module_cost_preload();
    let daemon = start_daemon().await;
    let dims = DIMS.to_string();
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        Some(&preload),
        None,
        &[
            ("SYNAPSE_TEST_DETERMINISTIC_DIMS", dims.as_str()),
            // No simulated engine latency: the measured remainder is module cost.
            ("SYNAPSE_TEST_DETERMINISTIC_DELAY_MS", "0"),
            // Every component non-zero, like a real embedding, so the JSON
            // reply has production size.
            ("SYNAPSE_TEST_DETERMINISTIC_DENSE", "1"),
            ("SYNAPSE_ANE_PROFILE_DIR", profile_dir.to_str().unwrap()),
        ],
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;
    certify_preloaded_models(&mut consumer, route, 70_000).await;

    let body = serde_json::json!({
        "method": "embed.batch",
        "params": { "model": "test-minilm", "items": rows() },
    });

    // The lane loads on first use; retry until it serves.
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut corr = 80_000;
    loop {
        let value = route_request(&mut consumer, route, corr, body.clone()).await;
        corr += 1;
        if value["result"]["error"]["code"] != "model_loading" {
            assert!(
                value["result"]["error"].is_null(),
                "first call failed: {value}"
            );
            break;
        }
        assert!(Instant::now() < deadline, "lane did not load: {value}");
        sleep(Duration::from_millis(100)).await;
    }
    let mut offsets = BTreeMap::new();
    // Warm-up calls (first-use allocation, lazy state) are discarded.
    for warmup in 0..2 {
        one_call(&mut consumer, route, 90_000 + warmup, &body).await;
    }
    read_new_stage_records(&profile_dir, &mut offsets);

    let mut samples = Vec::new();
    for run in 0..runs {
        let (mut sample, _) = one_call(&mut consumer, route, 91_000 + run as u64, &body).await;
        for record in read_new_stage_records(&profile_dir, &mut offsets) {
            let stage = record["stage"].as_str().unwrap().to_string();
            let entry = sample.stages.entry(stage).or_default();
            entry.0 += record["ms"].as_f64().unwrap();
            entry.1 += 1;
        }
        samples.push(sample);
    }

    let stage_names = samples
        .iter()
        .flat_map(|sample| sample.stages.keys().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let mut summary = serde_json::Map::new();
    let client = |pick: fn(&CallSample) -> f64| samples.iter().map(pick).collect::<Vec<_>>();
    summary.insert(
        "client_request_encode".into(),
        summarize("client_request_encode", client(|s| s.encode_ms)),
    );
    summary.insert(
        "client_roundtrip".into(),
        summarize("client_roundtrip", client(|s| s.roundtrip_ms)),
    );
    summary.insert(
        "client_reply_parse".into(),
        summarize("client_reply_parse", client(|s| s.parse_ms)),
    );
    summary.insert(
        "client_vector_extract".into(),
        summarize("client_vector_extract", client(|s| s.extract_ms)),
    );
    summary.insert(
        "client_total".into(),
        summarize(
            "client_total",
            client(|s| s.encode_ms + s.roundtrip_ms + s.parse_ms + s.extract_ms),
        ),
    );
    for stage in &stage_names {
        let values = samples
            .iter()
            .map(|sample| sample.stages.get(stage).map(|entry| entry.0).unwrap_or(0.0))
            .collect::<Vec<_>>();
        summary.insert(
            format!("module_{stage}"),
            summarize(&format!("module_{stage}"), values),
        );
    }
    let transport = samples
        .iter()
        .map(|sample| {
            sample.roundtrip_ms
                - sample
                    .stages
                    .get("handle_total")
                    .map(|entry| entry.0)
                    .unwrap_or(0.0)
        })
        .collect::<Vec<_>>();
    summary.insert(
        "transport_outside_handler".into(),
        summarize("transport_outside_handler", transport),
    );
    let qwen = match qwen_tokenizer() {
        Some(path) => {
            let (values, tokens) = qwen_tokenize_ms(&path, runs);
            eprintln!("module-cost qwen tokens for 64 code rows: {tokens}");
            serde_json::json!({
                "tokens": tokens,
                "summary": summarize("qwen_tokenize_batch", values),
            })
        }
        None => Value::Null,
    };
    let report = serde_json::json!({
        "rows": ROWS,
        "dims": DIMS,
        "runs": runs,
        "module_tokenizer": "fixture word-level, 120 words per row",
        "qwen_tokenize": qwen,
        "reply_bytes": samples.iter().map(|s| s.reply_bytes).collect::<Vec<_>>(),
        "total_tokens": samples.iter().map(|s| s.tokens).collect::<Vec<_>>(),
        "stage_counts": stage_names.iter().map(|stage| (stage.clone(), samples[0].stages.get(stage).map(|entry| entry.1).unwrap_or(0))).collect::<BTreeMap<_, _>>(),
        "summary": summary,
        "samples": samples.iter().map(|s| serde_json::json!({
            "client_request_encode_ms": s.encode_ms,
            "client_roundtrip_ms": s.roundtrip_ms,
            "client_reply_parse_ms": s.parse_ms,
            "client_vector_extract_ms": s.extract_ms,
            "module_stages_ms": s.stages.iter().map(|(k, v)| (k.clone(), v.0)).collect::<BTreeMap<_, _>>(),
        })).collect::<Vec<_>>(),
    });
    eprintln!(
        "module-cost reply_bytes={} tokens={}",
        samples[0].reply_bytes, samples[0].tokens
    );
    let path = out_dir.join("module-cost.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("module-cost report written to {}", path.display());
}
