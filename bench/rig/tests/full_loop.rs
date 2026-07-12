use std::fs;

use ahash::AHashMap;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

#[test]
fn rig_spawns_candidate_measures_gates_and_emits_result() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("synapse-rig-test-{}-{nonce}", std::process::id()));
    let model = root.join("model");
    fs::create_dir_all(&model).unwrap();
    fs::write(model.join("config.json"), r#"{"model_type":"bert"}"#).unwrap();

    let mut vocabulary = AHashMap::from_iter([("[UNK]".to_owned(), 0u32)]);
    for index in 0..12 {
        vocabulary.insert(format!("row{index}"), index + 1);
    }
    vocabulary.insert("fixture".to_owned(), 13);
    let word_level = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(word_level);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    let tokenizer_path = model.join("tokenizer.json");
    tokenizer.save(&tokenizer_path, false).unwrap();

    let corpus = root.join("corpus.jsonl");
    let reference = root.join("reference.jsonl");
    let mut corpus_rows = String::new();
    let mut reference_rows = String::new();
    for index in 0..12 {
        let id = format!("row-{index}");
        let text = format!("row{index} fixture");
        corpus_rows.push_str(&serde_json::json!({"id": id, "text": text}).to_string());
        corpus_rows.push('\n');
        reference_rows
            .push_str(&serde_json::json!({"id": id, "vec": fixture_vector(&text)}).to_string());
        reference_rows.push('\n');
    }
    fs::write(&corpus, corpus_rows).unwrap();
    fs::write(&reference, reference_rows).unwrap();
    let result_path = root.join("result.json");

    let output = Command::new(env!("CARGO_BIN_EXE_synapse-rig"))
        .arg("--candidate")
        .arg(env!("CARGO_BIN_EXE_rig-fixture-candidate"))
        .arg("--model")
        .arg(&model)
        .arg("--tokenizer")
        .arg(&tokenizer_path)
        .arg("--corpus")
        .arg(&corpus)
        .arg("--reference")
        .arg(&reference)
        .arg("--out")
        .arg(&result_path)
        .arg("--shapes")
        .arg("exact")
        .arg("--max-length")
        .arg("16")
        .arg("--attention-units")
        .arg("256")
        .arg("--passes")
        .arg("3")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "rig failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let result: serde_json::Value =
        serde_json::from_slice(&fs::read(&result_path).unwrap()).unwrap();
    assert_eq!(result["workload"], "embed-corpus-v1");
    assert_eq!(result["real_tokens"], 24);
    assert_eq!(result["passes"].as_array().unwrap().len(), 3);
    assert_eq!(result["passes"][0]["label"], "first");
    assert_eq!(result["passes"][1]["label"], "warm");
    assert_eq!(result["passes"][2]["label"], "steady");
    assert!(result["passes"][2]["parity_mean_cosine"].as_f64().unwrap() > 0.999_999);
    assert_eq!(result["passes"][2]["top10_rank_overlap"], 1.0);
    assert_eq!(result["rig_metadata"]["protocol_version"], 1);
    assert_eq!(
        result["rig_metadata"]["token_reconciliation"][2]["divergence_fraction"],
        0.0
    );
    assert_eq!(result["rig_metadata"]["sha256"].as_str().unwrap().len(), 64);
    assert!(!result["rig_metadata"]["git_revision"]
        .as_str()
        .unwrap()
        .is_empty());

    fs::remove_dir_all(root).unwrap();
}

fn fixture_vector(text: &str) -> Vec<f32> {
    let byte_sum = text.bytes().map(f32::from).sum::<f32>();
    let mut vector = vec![byte_sum + 1.0, text.len() as f32 + 1.0, 1.0];
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut vector {
        *value /= norm;
    }
    vector
}
