#![cfg(target_os = "macos")]

use std::{path::PathBuf, process::Command, time::SystemTime};

use synapse_module::worker_host::{WorkerEngine, WorkerHostConfig};

fn runtime_dir(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    PathBuf::from(format!("/tmp/sdw-{label}-{}-{suffix}", std::process::id()))
}

#[test]
fn fleet_binary_version_uses_decode_worker_name() {
    let output = Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-decode"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("ck-synapse-worker-decode "));
}

#[test]
fn worker_completes_standard_nonce_handshake_and_ping() {
    let runtime_dir = runtime_dir("ping");
    std::fs::create_dir_all(&runtime_dir).unwrap();
    let config =
        WorkerHostConfig::new(env!("CARGO_BIN_EXE_ck-synapse-worker-decode"), &runtime_dir);
    let engine = WorkerEngine::new(config).unwrap();
    let ping = engine.ping().unwrap();
    assert_eq!(ping.models_loaded, 0);
    drop(engine);
    let _ = std::fs::remove_dir_all(runtime_dir);
}

#[derive(serde::Deserialize)]
struct DecodePrompt {
    id: String,
    prompt: String,
}

#[derive(serde::Deserialize)]
struct ReferenceTokens {
    id: String,
    tokens: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct SpikeReference {
    results: Vec<ReferenceTokens>,
}

struct CheckpointFixture {
    model: PathBuf,
    tokenizer: PathBuf,
    prompts: Vec<DecodePrompt>,
    references: std::collections::BTreeMap<String, Vec<u32>>,
}

fn checkpoint_fixture() -> Option<CheckpointFixture> {
    let snapshot = std::env::var_os("SYNAPSE_OWNED_DECODE_QWEN3_0_6B").map(PathBuf::from)?;
    let model = snapshot.join("model.safetensors");
    let tokenizer = snapshot.join("tokenizer.json");
    if !model.is_file() || !tokenizer.is_file() {
        eprintln!(
            "skipping worker decode checkpoint e2e: {} is missing model.safetensors or tokenizer.json",
            snapshot.display()
        );
        return None;
    }
    let fixture_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/campaign/decode-fixtures");
    let prompts = std::fs::read_to_string(fixture_root.join("decode-prompts.jsonl"))
        .expect("read decode prompts")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("decode prompt row parses"))
        .collect::<Vec<_>>();
    let spike_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/spikes/unified-rt/fixtures/spike-qwen3-f16.jsonl");
    let spike_bytes = std::fs::read(&spike_path).expect("read pinned Qwen3 spike reference");
    assert_eq!(
        hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&spike_bytes)),
        "c3080813c45c364a73cbb6dce122afbba20e761b2189a31d0055ecf435232af1",
        "pinned Qwen3 engine fixture drifted"
    );
    let references: SpikeReference =
        serde_json::from_slice(&spike_bytes).expect("pinned Qwen3 spike reference parses");
    let references = references
        .results
        .into_iter()
        .map(|row| (row.id, row.tokens))
        .collect();
    Some(CheckpointFixture {
        model,
        tokenizer,
        prompts,
        references,
    })
}

fn sha256_file(path: &std::path::Path) -> String {
    use sha2::Digest;

    let bytes = std::fs::read(path).expect("read checkpoint artifact");
    hex::encode(sha2::Sha256::digest(bytes))
}

fn vocabulary_digest(tokenizer: &tokenizers::Tokenizer) -> String {
    use sha2::Digest;
    use synapse_engine_owned::owned_decode_engine::TokenVocabulary;

    let vocabulary = TokenVocabulary::from_tokenizer(tokenizer).expect("build token vocabulary");
    let mut hasher = sha2::Sha256::new();
    for token_id in 0..vocabulary.len() {
        if let Some(piece) = vocabulary.token_piece(token_id as u32) {
            hasher.update(piece);
        }
        hasher.update([0]);
    }
    hex::encode(hasher.finalize())
}

fn worker_constraint(
    compiled: &synapse_module::owned_decode_grammar_scheduler::TokenIdJsonConstraintV1,
) -> owned_decode_worker::protocol::TokenIdJsonConstraint {
    let runtime = &compiled.constraint_runtime_identity;
    owned_decode_worker::protocol::TokenIdJsonConstraint {
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

fn supervised_dispatch(
    fixture: &CheckpointFixture,
    label: &str,
    budget_path: &std::path::Path,
    prompt_ids: Vec<u32>,
    constraint: Option<owned_decode_worker::protocol::TokenIdJsonConstraint>,
    extra_args: Vec<String>,
) -> synapse_module::worker_host::SupervisedDecodeDispatch {
    use owned_decode_worker::{
        budget::BudgetPolicy,
        identity::QuarantineKey,
        protocol::{GenerateStart, Sampling},
        supervisor::TerminalControl,
        validation::WorkerStartContext,
    };
    use std::collections::BTreeMap;
    use std::time::Duration;
    use synapse_core::{RuntimeConfig, ValidatedArtifact};
    use synapse_module::worker_host::{OwnedDecodeWorkerFactory, SupervisedDecodeDispatch};

    let runtime_dir = runtime_dir(label);
    std::fs::create_dir_all(&runtime_dir).unwrap();
    let mut config =
        WorkerHostConfig::new(env!("CARGO_BIN_EXE_ck-synapse-worker-decode"), &runtime_dir);
    config.worker_id = label.to_string();
    config.request_timeout = Duration::from_secs(180);
    config.load_timeout = Duration::from_secs(180);
    config.extra_args = extra_args;
    let runtime_config_digest = "worker-e2e-runtime-v1".to_string();
    let decode_fingerprint = "worker-e2e-qwen3-f16".to_string();
    let mut values = BTreeMap::new();
    values.insert(
        "artifact_path".to_string(),
        fixture.model.to_string_lossy().to_string(),
    );
    values.insert("family".to_string(), "qwen3-0.6b".to_string());
    values.insert("weight_quant".to_string(), "f16".to_string());
    values.insert("context_bucket".to_string(), "512".to_string());
    values.insert("production_n".to_string(), "16".to_string());
    values.insert(
        "tokenizer_path".to_string(),
        fixture.tokenizer.to_string_lossy().to_string(),
    );
    values.insert("decode_fingerprint".to_string(), decode_fingerprint.clone());
    values.insert(
        "runtime_config_digest".to_string(),
        runtime_config_digest.clone(),
    );
    let artifact = ValidatedArtifact {
        digest: sha256_file(&fixture.model),
        format: "owned-safetensors".to_string(),
    };
    let factory = OwnedDecodeWorkerFactory::new(config, artifact, RuntimeConfig { values });
    let start = GenerateStart {
        generation_id: String::new(),
        loaded_model_ref: String::new(),
        decode_fingerprint: decode_fingerprint.clone(),
        runtime_config_digest: runtime_config_digest.clone(),
        prompt_ids,
        stop_ids: Vec::new(),
        max_tokens: 64,
        sampling: Sampling::greedy_top1(),
        constraint: constraint.clone(),
    };
    let context = WorkerStartContext {
        loaded_model_ref: String::new(),
        decode_fingerprint: decode_fingerprint.clone(),
        runtime_config_digest: runtime_config_digest.clone(),
        expected_constraint: constraint,
    };
    SupervisedDecodeDispatch::new(
        factory,
        budget_path,
        BudgetPolicy::default(),
        16,
        QuarantineKey::new(
            "worker-e2e-profile",
            decode_fingerprint,
            runtime_config_digest,
        ),
        start,
        context,
        TerminalControl::default(),
    )
    .expect("create supervised worker dispatch")
}

fn dispatch_command(
    generation_id: &str,
    prompt_token_count: usize,
    max_tokens: u32,
    constrained: bool,
) -> synapse_module::owned_decode_routing::DispatchedCommand {
    use synapse_core::Fingerprint;
    use synapse_module::owned_decode_routing::lane::LaneKind;

    synapse_module::owned_decode_routing::DispatchedCommand {
        lane: LaneKind::OwnedDecode,
        decode_fingerprint: Fingerprint("worker-e2e-qwen3-f16".to_string()),
        processing_fingerprint: Fingerprint("worker-e2e-processing-v1".to_string()),
        prompt_token_count: prompt_token_count.min(u32::MAX as usize) as u32,
        max_tokens,
        generation_id: generation_id.to_string(),
        constrained,
        chain_k: 1,
    }
}

#[test]
#[ignore]
fn checkpoint_worker_socket_is_token_exact_and_carries_compiled_constraint() {
    use synapse_module::owned_decode_grammar_scheduler::{
        compile_grammar, CompileContext, GrammarSubsetManifest,
    };
    use synapse_module::owned_decode_routing::DecodeDispatch;

    let Some(fixture) = checkpoint_fixture() else {
        eprintln!("skipping worker decode checkpoint e2e: set SYNAPSE_OWNED_DECODE_QWEN3_0_6B");
        return;
    };
    assert_eq!(fixture.prompts.len(), 20);
    let tokenizer = tokenizers::Tokenizer::from_file(&fixture.tokenizer).expect("load tokenizer");
    let budget_path = runtime_dir("battery-budget").join("budget.json");
    let first_prompt = tokenizer
        .encode(fixture.prompts[0].prompt.as_str(), true)
        .expect("tokenize first prompt")
        .get_ids()
        .to_vec();
    let mut dispatch = supervised_dispatch(
        &fixture,
        "battery",
        &budget_path,
        first_prompt.clone(),
        None,
        Vec::new(),
    );
    let mut worker_generation = None;
    for (index, prompt) in fixture.prompts.iter().enumerate() {
        let prompt_ids = tokenizer
            .encode(prompt.prompt.as_str(), true)
            .expect("tokenize fixture prompt")
            .get_ids()
            .to_vec();
        dispatch.set_request(prompt_ids.clone(), None, 180_000);
        let response = dispatch
            .dispatch(&dispatch_command(
                &format!("battery-{index}"),
                prompt_ids.len(),
                64,
                false,
            ))
            .expect("worker fixture generation succeeds");
        match worker_generation {
            Some(generation) => assert_eq!(response.worker_generation, generation),
            None => worker_generation = Some(response.worker_generation),
        }
        assert_eq!(
            response.generated_token_ids,
            *fixture
                .references
                .get(&prompt.id)
                .expect("reference exists"),
            "worker socket diverged for {}",
            prompt.id
        );
    }

    let schema = r#"{"type":"null"}"#;
    let compiled = compile_grammar(
        schema,
        &CompileContext {
            base_decode_fingerprint: synapse_core::Fingerprint("worker-e2e-qwen3-f16".to_string()),
            tokenizer_vocabulary_digest: vocabulary_digest(&tokenizer),
        },
        &GrammarSubsetManifest::default(),
    )
    .expect("compile constraint");
    let constrained_prompt = tokenizer
        .encode(
            "Respond with exactly the JSON literal null and nothing else:\n",
            true,
        )
        .expect("tokenize constrained prompt")
        .get_ids()
        .to_vec();
    dispatch.set_request(
        constrained_prompt.clone(),
        Some(worker_constraint(&compiled.constraint)),
        180_000,
    );
    let response = dispatch
        .dispatch(&dispatch_command(
            "constrained",
            constrained_prompt.len(),
            64,
            true,
        ))
        .expect("constrained worker generation succeeds");
    let text = tokenizer
        .decode(&response.generated_token_ids, true)
        .expect("decode constrained output");
    let value: serde_json::Value = serde_json::from_str(&text).expect("output is valid JSON");
    assert!(value.is_null());
}

#[test]
#[ignore]
fn checkpoint_worker_crash_redispatches_once_then_quarantines() {
    use synapse_module::owned_decode_routing::{error::OwnedDecodeError, DecodeDispatch};

    let Some(fixture) = checkpoint_fixture() else {
        eprintln!("skipping worker crash checkpoint e2e: set SYNAPSE_OWNED_DECODE_QWEN3_0_6B");
        return;
    };
    let tokenizer = tokenizers::Tokenizer::from_file(&fixture.tokenizer).expect("load tokenizer");
    let prompt_ids = tokenizer
        .encode(fixture.prompts[0].prompt.as_str(), true)
        .expect("tokenize crash prompt")
        .get_ids()
        .to_vec();
    let state_dir = runtime_dir("crash-state");
    std::fs::create_dir_all(&state_dir).unwrap();
    let budget_path = state_dir.join("budget.json");
    let marker = state_dir.join("crashed-once");
    let mut once = supervised_dispatch(
        &fixture,
        "crash-once",
        &budget_path,
        prompt_ids.clone(),
        None,
        vec![
            "--test-abort-after-progress-once".to_string(),
            marker.to_string_lossy().to_string(),
        ],
    );
    let recovered = once
        .dispatch(&dispatch_command("crash-once", prompt_ids.len(), 64, false))
        .expect("one crash is deterministically redispatched");
    assert_eq!(
        recovered.generated_token_ids,
        *fixture
            .references
            .get(&fixture.prompts[0].id)
            .expect("crash prompt reference exists"),
        "redispatch must deterministically re-execute the prompt"
    );
    assert_eq!(recovered.crash_retry_count, 1);
    assert_eq!(recovered.failure_classifications, ["crash"]);
    assert_eq!(once.crash_budget_remaining(), 1);

    let mut exhaust = supervised_dispatch(
        &fixture,
        "crash-exhaust",
        &budget_path,
        prompt_ids.clone(),
        None,
        vec!["--test-abort-after-progress".to_string()],
    );
    assert_eq!(
        exhaust.dispatch(&dispatch_command(
            "crash-exhaust",
            prompt_ids.len(),
            64,
            false,
        )),
        Err(OwnedDecodeError::Quarantined)
    );
    assert!(exhaust.is_quarantined());
    assert_eq!(exhaust.crash_budget_remaining(), 0);
    assert_eq!(
        exhaust.dispatch(&dispatch_command(
            "quarantine-refusal",
            prompt_ids.len(),
            64,
            false,
        )),
        Err(OwnedDecodeError::Quarantined),
        "an exhausted key refuses before spawning another worker"
    );
}
