#![cfg(target_os = "macos")]

use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    net::Shutdown,
    os::unix::net::{UnixListener, UnixStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime},
};

use owned_decode_worker::{
    identity::WORKER_PROTOCOL_ID,
    protocol::{
        DecodeTransportRequest, DecodeTransportResponse, FinalResponse, FrameEnvelope,
        GenerateContinue, GenerateInstallHintBank, GenerateStart, HintVerificationStats, Sampling,
        TokenIdJsonConstraint, WorkerFrame,
    },
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use synapse_core::{
    worker_framing_sync::{read_frame, write_json_frame},
    worker_protocol::{
        WorkerHello, WorkerHelloAck, WorkerRequest, WorkerResponse, DEFAULT_MAX_FRAME_BYTES,
        WORKER_PROTOCOL_VERSION,
    },
    Fingerprint, SidecarHintBank,
};
use synapse_engine_owned::owned_decode_engine::TokenVocabulary;
use synapse_module::owned_decode_grammar_scheduler::{
    compile_grammar, grammar_automaton::Automaton, CompileContext, GrammarSubsetManifest,
    TokenIdJsonConstraintV1,
};
use tokenizers::Tokenizer;

const BOUNDARY: &str =
    "handler_entry_after_arm_identified_to_response_ready_after_cancellation_signal";
const SOURCE_MODEL_DIGEST: &str =
    "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd";
const DERIVED_Q8_DIGEST: &str = "17d2fbfeff90269190287f324ed93bab3bb1b4fa4aad98c3fbba1868c01cb0f2";
const RENDER_POLICY_DIGEST: &str =
    "5e34db7ec9bc172e3088606bf8a1810dd0cdbe5fc083e323ba2f1ead2cd690c4";
const AUTOMATON_STATE_VISIT_CAP: usize = 4096;
const REPETITIONS: u32 = 5;
const WARMUPS: u32 = 5;
const ARM_ORDER_SEED: u64 = 20260810;
const DECODE_FINGERPRINT: &str = "semantic-sidecar-phase1-qwen3-0.6b-q8_0";
const RUNTIME_CONFIG_DIGEST: &str =
    "17d2fbfeff90269190287f324ed93bab3bb1b4fa4aad98c3fbba1868c01cb0f2";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    B0,
    B1,
    SoBank,
}

impl Arm {
    const ALL: [Self; 3] = [Self::B0, Self::B1, Self::SoBank];

    const fn id(self) -> &'static str {
        match self {
            Self::B0 => "B0",
            Self::B1 => "B1",
            Self::SoBank => "SO-BANK",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct CohortRow {
    request_id: String,
    #[serde(default)]
    adversarial_schema_id: Option<String>,
    prompt: String,
    grammar: String,
    max_tokens: u32,
}

#[derive(Clone)]
struct PreparedRequest {
    partition: &'static str,
    workload: &'static str,
    row: CohortRow,
    prompt_ids: Vec<u32>,
    constraint: TokenIdJsonConstraint,
    bank: SidecarHintBank,
    automaton_prefix_trace_digest: String,
    automaton_state_visits: usize,
}

#[derive(Clone, Debug, Serialize)]
struct MachineState {
    hardware_model: String,
    os_version: String,
    load_average_at_start: f64,
    thermal_snapshot: String,
    worker_binary: String,
    source_model_sha256: String,
    derived_q8_sha256: String,
}

struct Generated {
    final_response: FinalResponse,
    wall_clock_ms: f64,
}

struct WorkerClient {
    child: Child,
    stream: UnixStream,
    model_ref: String,
    request_counter: u64,
    socket_path: PathBuf,
}

impl WorkerClient {
    fn spawn(snapshot: &Path, arm: Arm) -> Self {
        let runtime_dir = runtime_dir(arm.id());
        fs::create_dir_all(&runtime_dir).expect("create phase-1 worker runtime directory");
        let socket_path = runtime_dir.join("worker.sock");
        let listener = UnixListener::bind(&socket_path).expect("bind phase-1 worker socket");
        let nonce = format!(
            "{:016x}",
            stable_u64(format!("{}-{}", arm.id(), std::process::id()).as_bytes())
        );
        let mut command = Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-decode"));
        command
            .arg("--socket")
            .arg(&socket_path)
            .arg("--nonce")
            .arg(&nonce)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        if arm == Arm::B0 {
            command.arg("--disable-forced-token-fast-path");
        }
        let child = command.spawn().expect("spawn real phase-1 worker binary");
        let (mut stream, _) = listener.accept().expect("accept phase-1 worker connection");
        stream
            .set_read_timeout(Some(Duration::from_secs(600)))
            .expect("set worker read timeout");
        stream
            .set_write_timeout(Some(Duration::from_secs(600)))
            .expect("set worker write timeout");
        let hello: WorkerHello = read_json(&mut stream);
        assert_eq!(hello.v, WORKER_PROTOCOL_VERSION);
        assert_eq!(hello.nonce, nonce);
        let max_frame = hello.max_frame.min(DEFAULT_MAX_FRAME_BYTES);
        write_json_frame(
            &mut stream,
            &WorkerHelloAck {
                v: WORKER_PROTOCOL_VERSION,
                accept: true,
                max_frame,
            },
            max_frame,
        )
        .expect("acknowledge phase-1 worker");

        let model_path = snapshot.join("model.safetensors");
        let tokenizer_path = snapshot.join("tokenizer.json");
        let mut runtime_config = BTreeMap::new();
        runtime_config.insert(
            "artifact_path".to_string(),
            model_path.display().to_string(),
        );
        runtime_config.insert("family".to_string(), "qwen3-0.6b".to_string());
        runtime_config.insert("weight_quant".to_string(), "q8_0".to_string());
        runtime_config.insert("context_bucket".to_string(), "2048".to_string());
        runtime_config.insert("production_n".to_string(), "16".to_string());
        runtime_config.insert(
            "tokenizer_path".to_string(),
            tokenizer_path.display().to_string(),
        );
        runtime_config.insert(
            "decode_fingerprint".to_string(),
            DECODE_FINGERPRINT.to_string(),
        );
        runtime_config.insert(
            "runtime_config_digest".to_string(),
            RUNTIME_CONFIG_DIGEST.to_string(),
        );
        let load = WorkerRequest::Load {
            req_id: "phase1-load".to_string(),
            artifact_path: model_path.display().to_string(),
            artifact_digest: SOURCE_MODEL_DIGEST.to_string(),
            format: "owned-safetensors".to_string(),
            runtime_config,
        };
        write_json_frame(&mut stream, &load, max_frame).expect("send phase-1 model load");
        let response: WorkerResponse = read_json_with_limit(&mut stream, max_frame);
        let model_ref = match response {
            WorkerResponse::Loaded { model_ref, .. } => model_ref,
            other => panic!("phase-1 worker load failed: {other:?}"),
        };
        Self {
            child,
            stream,
            model_ref,
            request_counter: 0,
            socket_path,
        }
    }

    fn request(&mut self, request: DecodeTransportRequest) -> DecodeTransportResponse {
        write_json_frame(&mut self.stream, &request, DEFAULT_MAX_FRAME_BYTES)
            .expect("send owned-decode phase-1 request");
        read_json(&mut self.stream)
    }

    fn generate(
        &mut self,
        request: &PreparedRequest,
        arm: Arm,
        generation_id: String,
    ) -> Generated {
        let started = Instant::now();
        let start_req_id = self.next_req_id("start");
        let response = self.request(DecodeTransportRequest::GenerateStart {
            req_id: start_req_id,
            start: Box::new(GenerateStart {
                generation_id: generation_id.clone(),
                loaded_model_ref: self.model_ref.clone(),
                decode_fingerprint: DECODE_FINGERPRINT.to_string(),
                runtime_config_digest: RUNTIME_CONFIG_DIGEST.to_string(),
                prompt_ids: request.prompt_ids.clone(),
                stop_ids: Vec::new(),
                max_tokens: request.row.max_tokens,
                sampling: Sampling::greedy_top1(),
                constraint: Some(request.constraint.clone()),
            }),
        });
        let mut envelope = response_frame(response, "GENERATE_START");
        let mut bank_installed = false;
        loop {
            assert_eq!(envelope.protocol, WORKER_PROTOCOL_ID);
            match envelope.frame {
                WorkerFrame::Progress(progress) => {
                    if arm == Arm::SoBank && !bank_installed {
                        let bank_req_id = self.next_req_id("bank");
                        let response =
                            self.request(DecodeTransportRequest::GenerateInstallHintBank {
                                req_id: bank_req_id,
                                installation: GenerateInstallHintBank {
                                    generation_id: generation_id.clone(),
                                    bank: request.bank.clone(),
                                },
                            });
                        match response {
                            DecodeTransportResponse::HintBankInstalled { installation, .. } => {
                                assert_eq!(installation.generation_id, generation_id);
                                assert_eq!(
                                    installation.bank_content_digest,
                                    request.bank.content_digest()
                                );
                            }
                            other => panic!("SO-BANK installation failed: {other:?}"),
                        }
                        bank_installed = true;
                    }
                    let remaining = request
                        .row
                        .max_tokens
                        .saturating_sub(progress.committed_token_count);
                    assert!(
                        remaining > 0,
                        "progress consumed the complete request budget"
                    );
                    let continue_req_id = self.next_req_id("continue");
                    envelope = response_frame(
                        self.request(DecodeTransportRequest::GenerateContinue {
                            req_id: continue_req_id,
                            continuation: GenerateContinue {
                                generation_id: generation_id.clone(),
                                next_expected_sequence: progress.quantum_sequence + 1,
                                next_token_budget: remaining.min(16),
                            },
                        }),
                        "GENERATE_CONTINUE",
                    );
                }
                WorkerFrame::Final(final_response) => {
                    // Deterministic arms do not launch asynchronous work. Their
                    // cancellation signal is therefore an immediate no-op at the
                    // same response-ready boundary used by sidecar arms.
                    let wall_clock_ms = started.elapsed().as_secs_f64() * 1000.0;
                    return Generated {
                        final_response,
                        wall_clock_ms,
                    };
                }
                WorkerFrame::Error { id } => panic!("phase-1 generation failed with {id}"),
            }
        }
    }

    fn next_req_id(&mut self, prefix: &str) -> String {
        self.request_counter += 1;
        format!("phase1-{prefix}-{}", self.request_counter)
    }
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        let _ = write_json_frame(
            &mut self.stream,
            &WorkerRequest::Shutdown {},
            DEFAULT_MAX_FRAME_BYTES,
        );
        let _ = self.stream.shutdown(Shutdown::Both);
        let _ = self.child.wait();
        let _ = fs::remove_file(&self.socket_path);
        if let Some(parent) = self.socket_path.parent() {
            let _ = fs::remove_dir(parent);
        }
    }
}

#[test]
#[ignore]
fn measure_semantic_sidecar_deterministic_phase1() {
    if std::env::var_os("SYNAPSE_RUN_SEMANTIC_SIDECAR_PHASE1").is_none() {
        eprintln!("skipping phase-1 measurement; set SYNAPSE_RUN_SEMANTIC_SIDECAR_PHASE1=1");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let evidence = root.join("evidence/semantic-sidecar-v1");
    let snapshot = PathBuf::from(
        std::env::var_os("SYNAPSE_OWNED_DECODE_QWEN3_0_6B")
            .expect("set SYNAPSE_OWNED_DECODE_QWEN3_0_6B to the pinned snapshot directory"),
    );
    assert_eq!(
        sha256_file(&snapshot.join("model.safetensors")),
        SOURCE_MODEL_DIGEST
    );
    let machine_state = capture_machine_state();
    let smoke_count = std::env::var("SYNAPSE_SEMANTIC_SIDECAR_SMOKE_REQUESTS")
        .ok()
        .map(|value| value.parse::<usize>().expect("parse smoke request count"));
    if smoke_count.is_none() {
        assert!(
            start_load_is_eligible(machine_state.load_average_at_start),
            "refusing phase-1 measurement at load average {:.2}; contract maximum is 8",
            machine_state.load_average_at_start
        );
    }

    let tokenizer =
        Tokenizer::from_file(snapshot.join("tokenizer.json")).expect("load target tokenizer");
    let vocabulary_digest = vocabulary_digest(&tokenizer);
    let mut requests = prepare_requests(&evidence, &tokenizer, &vocabulary_digest);
    if let Some(limit) = smoke_count {
        requests.truncate(limit.max(1));
    }
    let output_path = std::env::var_os("SYNAPSE_SEMANTIC_SIDECAR_PHASE1_ROWS")
        .map(PathBuf::from)
        .unwrap_or_else(|| evidence.join("measurement-rows-phase1-v1.jsonl"));
    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&output_path)
        .expect("create phase-1 measurement rows");

    let mut clients = BTreeMap::new();
    clients.insert(Arm::B0.id(), WorkerClient::spawn(&snapshot, Arm::B0));
    clients.insert(Arm::B1.id(), WorkerClient::spawn(&snapshot, Arm::B1));
    clients.insert(
        Arm::SoBank.id(),
        WorkerClient::spawn(&snapshot, Arm::SoBank),
    );

    for (request_index, request) in requests.iter().enumerate() {
        let mut exact_output: Option<(Vec<u32>, String, String)> = None;
        for warmup in 1..=WARMUPS {
            for arm in randomized_arms(&request.row.request_id, "warmup", warmup) {
                let generated = clients
                    .get_mut(arm.id())
                    .expect("arm worker exists")
                    .generate(
                        request,
                        arm,
                        format!("{}-warmup-{warmup}-{}", request.row.request_id, arm.id()),
                    );
                enforce_exactness(
                    request,
                    arm,
                    &generated.final_response,
                    &tokenizer,
                    &mut exact_output,
                );
            }
        }
        for repetition in 1..=REPETITIONS {
            for arm in randomized_arms(&request.row.request_id, "measured", repetition) {
                let generated = clients
                    .get_mut(arm.id())
                    .expect("arm worker exists")
                    .generate(
                        request,
                        arm,
                        format!(
                            "{}-measured-{repetition}-{}",
                            request.row.request_id,
                            arm.id()
                        ),
                    );
                let output_identity = output_identity(&generated.final_response, &tokenizer);
                let row = measurement_row(
                    request,
                    arm,
                    repetition,
                    &generated,
                    &output_identity,
                    &machine_state,
                    &vocabulary_digest,
                );
                serde_json::to_writer(&mut output, &row).expect("serialize phase-1 row");
                output.write_all(b"\n").expect("append phase-1 newline");
                output.flush().expect("preserve phase-1 row");
                enforce_exactness(
                    request,
                    arm,
                    &generated.final_response,
                    &tokenizer,
                    &mut exact_output,
                );
            }
        }
        eprintln!(
            "phase1 completed {}/{}: {}/{}",
            request_index + 1,
            requests.len(),
            request.partition,
            request.row.request_id
        );
    }
}

fn prepare_requests(
    evidence: &Path,
    tokenizer: &Tokenizer,
    vocabulary_digest: &str,
) -> Vec<PreparedRequest> {
    let mut requests = Vec::new();
    for (file, partition) in [
        ("cohort-classify-v1.jsonl", "primary"),
        ("cohort-adversarial-v1.jsonl", "adversarial"),
    ] {
        for row in read_jsonl::<CohortRow>(&evidence.join(file)) {
            let prompt_ids = tokenizer
                .encode(row.prompt.as_str(), true)
                .expect("tokenize cohort prompt")
                .get_ids()
                .to_vec();
            assert!(
                prompt_ids.len() + row.max_tokens as usize <= 2048,
                "cohort request exceeds the frozen context bucket: {}",
                row.request_id
            );
            let compiled = compile_grammar(
                &row.grammar,
                &CompileContext {
                    base_decode_fingerprint: Fingerprint(DECODE_FINGERPRINT.to_string()),
                    tokenizer_vocabulary_digest: vocabulary_digest.to_string(),
                },
                &GrammarSubsetManifest::default(),
            )
            .expect("compile cohort grammar");
            let (rendered, trace_digest, state_visits) =
                deterministic_completion(&compiled.automaton, &row.grammar);
            let bank_ids = tokenizer
                .encode(rendered, false)
                .expect("target-tokenize deterministic SO-BANK view")
                .get_ids()
                .to_vec();
            assert!(!bank_ids.is_empty() && bank_ids.len() <= 4096);
            let bank = SidecarHintBank {
                views: vec![bank_ids],
                schema_identity: compiled.constraint.canonical_schema_digest.clone(),
                render_policy_digest: RENDER_POLICY_DIGEST.to_string(),
                built_at: 0,
            };
            requests.push(PreparedRequest {
                partition,
                workload: "athena-classify-json",
                row,
                prompt_ids,
                constraint: worker_constraint(&compiled.constraint),
                bank,
                automaton_prefix_trace_digest: trace_digest,
                automaton_state_visits: state_visits,
            });
        }
    }
    requests
}

fn deterministic_completion(automaton: &Automaton, raw_schema: &str) -> (String, String, usize) {
    let schema: serde_json::Value = serde_json::from_str(raw_schema).expect("parse cohort schema");
    let rendered = serde_json::to_string(&canonical_schema_value(&schema))
        .expect("render compact canonical schema value");
    let mut state = automaton.initial();
    let mut trace = Vec::<(usize, usize, u8)>::new();
    for (offset, byte) in rendered.bytes().enumerate() {
        assert!(
            offset < AUTOMATON_STATE_VISIT_CAP,
            "SO-BANK automaton visit cap exceeded"
        );
        let permitted = automaton
            .permitted_bytes(&state)
            .into_iter()
            .filter(|candidate| !matches!(candidate, b' ' | b'\t' | b'\n' | b'\r'))
            .collect::<Vec<_>>();
        assert!(
            permitted.contains(&byte),
            "canonical renderer emitted an illegal byte"
        );
        trace.push((offset, permitted.len(), byte));
        state = automaton
            .step(&state, byte)
            .expect("advance canonical SO-BANK trace");
    }
    assert!(
        automaton.has_complete_value(&state),
        "canonical SO-BANK render is incomplete"
    );
    let visits = trace.len();
    let trace_bytes = serde_json::to_vec(&trace).expect("serialize automaton prefix trace");
    (rendered, sha256_bytes(&trace_bytes), visits)
}

fn canonical_schema_value(schema: &serde_json::Value) -> serde_json::Value {
    let ty = schema
        .get("type")
        .and_then(serde_json::Value::as_str)
        .expect("schema node has a type");
    if let Some(values) = schema.get("enum").and_then(serde_json::Value::as_array) {
        return values
            .iter()
            .min_by_key(|value| serde_json::to_string(value).expect("serialize enum value"))
            .expect("schema enum is non-empty")
            .clone();
    }
    match ty {
        "object" => {
            let properties = schema
                .get("properties")
                .and_then(serde_json::Value::as_object)
                .expect("object schema has properties");
            let required = schema
                .get("required")
                .and_then(serde_json::Value::as_array)
                .expect("object schema has required properties");
            let mut value = serde_json::Map::new();
            for name in required
                .iter()
                .map(|name| name.as_str().expect("required property is a string"))
                .collect::<std::collections::BTreeSet<_>>()
            {
                value.insert(
                    name.to_string(),
                    canonical_schema_value(
                        properties.get(name).expect("required property is declared"),
                    ),
                );
            }
            serde_json::Value::Object(value)
        }
        "array" => serde_json::json!([]),
        "string" => serde_json::json!(""),
        "number" | "integer" => serde_json::json!(0),
        "boolean" => serde_json::json!(false),
        "null" => serde_json::Value::Null,
        other => panic!("unsupported canonical schema type {other}"),
    }
}

fn measurement_row(
    request: &PreparedRequest,
    arm: Arm,
    repetition: u32,
    generated: &Generated,
    output_identity: &(Vec<u32>, String, String),
    machine_state: &MachineState,
    vocabulary_digest: &str,
) -> serde_json::Value {
    let (ids, decoded_digest, finish_reason) = output_identity;
    let mut row = serde_json::json!({
        "partition": request.partition,
        "workload": request.workload,
        "request_id": request.row.request_id,
        "arm_id": arm.id(),
        "repetition": repetition,
        "wall_clock_boundary": BOUNDARY,
        "wall_clock_ms": generated.wall_clock_ms,
        "generated_token_ids": ids,
        "decoded_response_sha256": decoded_digest,
        "finish_reason": finish_reason,
        "instrumentation": {
            "target_decode_ms": generated.wall_clock_ms,
            "response_ready_ms": generated.wall_clock_ms,
        },
        "machine_state": machine_state,
        "thermal_policy": {"maximum_start_load_average": 8.0, "sequential": true},
        "deadline_and_queue_timeout": {"worker_io_timeout_ms": 600000, "queue_timeout_ms": 0},
    });
    if let Some(schema_id) = &request.row.adversarial_schema_id {
        row["adversarial_schema_id"] = serde_json::json!(schema_id);
    }
    if arm == Arm::SoBank {
        row["so_bank"] = serde_json::json!({
            "source": "longest_common_legal_completion_byte_prefix",
            "automaton_state_visit_cap": AUTOMATON_STATE_VISIT_CAP,
            "automaton_state_visits": request.automaton_state_visits,
            "automaton_prefix_trace_digest": request.automaton_prefix_trace_digest,
            "max_suffix_match_tokens": 7,
            "max_proposal_tokens": 16,
            "content_digest": request.bank.content_digest(),
            "target_tokenizer_digest": vocabulary_digest,
            "render_policy_digest": RENDER_POLICY_DIGEST,
        });
        row["verification"] = verification_json(&generated.final_response.hint_verification);
    }
    row
}

fn verification_json(stats: &HintVerificationStats) -> serde_json::Value {
    serde_json::json!({
        "grammar_masked": true,
        "proposed_tokens": stats.proposed_tokens,
        "verified_tokens": stats.verified_tokens,
        "accepted_tokens": stats.accepted_tokens,
        "rejected_proposal_attempts": stats.rejected_proposal_attempts,
        "accepted_tokens_by_span": stats.accepted_tokens_by_span,
        "first_divergence_categories": stats.first_divergence_categories,
    })
}

fn enforce_exactness(
    request: &PreparedRequest,
    arm: Arm,
    response: &FinalResponse,
    tokenizer: &Tokenizer,
    expected: &mut Option<(Vec<u32>, String, String)>,
) {
    let actual = output_identity(response, tokenizer);
    if let Some(expected) = expected {
        assert_eq!(
            &actual,
            expected,
            "exactness divergence for {}/{} in {}",
            request.partition,
            request.row.request_id,
            arm.id()
        );
    } else {
        *expected = Some(actual);
    }
}

fn output_identity(response: &FinalResponse, tokenizer: &Tokenizer) -> (Vec<u32>, String, String) {
    let decoded = tokenizer
        .decode(&response.generated_ids, true)
        .expect("decode generated target IDs");
    let finish = serde_json::to_value(response.finish_reason)
        .expect("serialize finish reason")
        .as_str()
        .expect("finish reason serializes as string")
        .to_string();
    (
        response.generated_ids.clone(),
        sha256_bytes(decoded.as_bytes()),
        finish,
    )
}

fn randomized_arms(request_id: &str, phase: &str, block: u32) -> Vec<Arm> {
    let mut arms = Arm::ALL.to_vec();
    arms.sort_by_key(|arm| {
        stable_u64(
            format!(
                "{ARM_ORDER_SEED}\n{request_id}\n{phase}\n{block}\n{}",
                arm.id()
            )
            .as_bytes(),
        )
    });
    arms
}

const fn start_load_is_eligible(load_average: f64) -> bool {
    load_average <= 8.0
}

#[test]
fn thermal_policy_refuses_load_above_contract_limit() {
    assert!(start_load_is_eligible(8.0));
    assert!(!start_load_is_eligible(8.01));
}

#[test]
fn arm_order_is_seeded_and_covers_each_arm_once() {
    let first = randomized_arms("request-1", "measured", 1);
    let repeated = randomized_arms("request-1", "measured", 1);
    assert_eq!(first, repeated);
    assert_eq!(first.len(), Arm::ALL.len());
    for arm in Arm::ALL {
        assert_eq!(
            first.iter().filter(|candidate| **candidate == arm).count(),
            1
        );
    }
}

fn capture_machine_state() -> MachineState {
    let load_average_at_start = command_output("sysctl", &["-n", "vm.loadavg"])
        .split(|character: char| !(character.is_ascii_digit() || character == '.'))
        .find_map(|part| part.parse::<f64>().ok())
        .expect("read load average from sysctl");
    MachineState {
        hardware_model: command_output("sysctl", &["-n", "hw.model"]),
        os_version: command_output("sw_vers", &["-productVersion"]),
        load_average_at_start,
        thermal_snapshot: command_output("pmset", &["-g", "therm"]),
        worker_binary: env!("CARGO_BIN_EXE_ck-synapse-worker-decode").to_string(),
        source_model_sha256: SOURCE_MODEL_DIGEST.to_string(),
        derived_q8_sha256: DERIVED_Q8_DIGEST.to_string(),
    }
}

fn command_output(program: &str, args: &[&str]) -> String {
    let output = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("run {program}: {error}"));
    assert!(output.status.success(), "{program} failed");
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn worker_constraint(compiled: &TokenIdJsonConstraintV1) -> TokenIdJsonConstraint {
    let runtime = &compiled.constraint_runtime_identity;
    TokenIdJsonConstraint {
        encoding_id: compiled.representation_revision.clone(),
        constraint_runtime_identity: runtime.digest(),
        constraint_fingerprint: compiled.constraint_fingerprint.0.clone(),
        grammar_subset_revision: runtime.grammar_subset_revision.clone(),
        grammar_compiler_revision: runtime.grammar_compiler_revision.clone(),
        tokenizer_vocabulary_digest: compiled.tokenizer_vocabulary_digest.clone(),
        limits_manifest_id: compiled.limits_manifest_id.clone(),
        worker_constraint_runtime_revision: runtime.worker_constraint_runtime_revision.clone(),
        canonical_schema_digest: compiled.canonical_schema_digest.clone(),
        initial_state_encoding: compiled.initial_state_encoding.clone(),
        initial_state_digest: compiled.initial_state_digest.clone(),
        compiled_automaton_digest: compiled.compiled_automaton_digest.clone(),
        automaton_bytes: compiled.automaton_bytes.clone(),
    }
}

fn response_frame(response: DecodeTransportResponse, operation: &str) -> FrameEnvelope {
    match response {
        DecodeTransportResponse::Frame { envelope, .. } => envelope,
        other => panic!("{operation} returned unexpected response {other:?}"),
    }
}

fn read_json<T: serde::de::DeserializeOwned>(stream: &mut UnixStream) -> T {
    read_json_with_limit(stream, DEFAULT_MAX_FRAME_BYTES)
}

fn read_json_with_limit<T: serde::de::DeserializeOwned>(
    stream: &mut UnixStream,
    max_frame: u32,
) -> T {
    let bytes = read_frame(stream, max_frame).expect("read worker JSON frame");
    serde_json::from_slice(&bytes).expect("decode worker JSON frame")
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Vec<T> {
    let mut text = String::new();
    File::open(path)
        .and_then(|mut file| file.read_to_string(&mut text))
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("decode cohort row"))
        .collect()
}

fn vocabulary_digest(tokenizer: &Tokenizer) -> String {
    let vocabulary = TokenVocabulary::from_tokenizer(tokenizer).expect("build target vocabulary");
    let mut hasher = Sha256::new();
    for token_id in 0..vocabulary.len() {
        if let Some(piece) = vocabulary.token_piece(token_id as u32) {
            hasher.update(piece);
        }
        hasher.update([0]);
    }
    hex::encode(hasher.finalize())
}

fn runtime_dir(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock is after epoch")
        .as_nanos();
    PathBuf::from(format!(
        "/tmp/sdw-phase1-{label}-{}-{suffix}",
        std::process::id()
    ))
}

fn sha256_file(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    sha256_bytes(&bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn stable_u64(bytes: &[u8]) -> u64 {
    let digest = Sha256::digest(bytes);
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}
