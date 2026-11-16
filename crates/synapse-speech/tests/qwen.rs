use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use serde_json::Value;
use sha2::{Digest, Sha256};
use synapse_speech::qwen;

fn parity() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity")
}
fn json(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn corpus() -> Value {
    json(parity().join("reference/qwen3-asr-corpus.json"))
}
fn trace(id: &str) -> Value {
    static REFERENCE: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    let reference =
        REFERENCE.get_or_init(|| json(parity().join("fixtures/qwen3-asr-1.7b/reference.json")));
    let seal = &reference["cases"][id];
    let path = parity()
        .join("fixtures/qwen3-asr-1.7b")
        .join(seal["path"].as_str().unwrap());
    let raw = fs::read(path).unwrap();
    assert_eq!(raw.len() as u64, seal["bytes"].as_u64().unwrap(), "{id}");
    assert_eq!(sha(&raw), seal["sha256"].as_str().unwrap(), "{id}");
    serde_json::from_slice(&raw).unwrap()
}

#[test]
fn qwen_frame_and_chunk_boundaries() {
    for (frames, tokens) in [
        (0, 0),
        (1, 1),
        (8, 1),
        (9, 2),
        (99, 13),
        (100, 13),
        (101, 14),
        (137, 18),
        (199, 26),
        (200, 26),
        (60000, 7800),
    ] {
        assert_eq!(qwen::audio_tokens(frames), tokens);
    }
    for (samples, frames) in [
        (201, 1),
        (1599, 9),
        (1600, 10),
        (21920, 137),
        (501920, 3137),
    ] {
        assert_eq!(qwen::mel_frames(samples), frames);
    }
    let frontend = qwen::Frontend::new();
    let silence = frontend.process_samples(&vec![0.0; 21921]).unwrap();
    assert_eq!(silence.mel_frames, 137);
    assert_eq!(silence.valid_mel_frames, 137);
    assert_eq!(silence.audio_tokens, 18);
    assert!(silence.values.iter().all(|&v| v == -1.5));
    for samples in [
        vec![],
        vec![0.0; 200],
        vec![f32::NAN; 201],
        vec![f32::MAX; 201],
    ] {
        assert!(frontend.process_samples(&samples).is_err());
    }
}

#[test]
fn qwen_frontend_reference_edges() {
    let frontend = qwen::Frontend::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen-frontend");
    for id in ["frontend-short", "frontend-sub-100-remainder"] {
        let trace = trace(id);
        let pcm = fs::read(root.join(format!("{id}.pcm"))).unwrap();
        let mel = fs::read(root.join(format!("{id}.mel"))).unwrap();
        assert_eq!(
            sha(&pcm),
            trace["model_input_audio_sha256"].as_str().unwrap()
        );
        assert_eq!(sha(&mel), trace["mel"]["sha256"].as_str().unwrap());
        let samples: Vec<_> = pcm
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let features = frontend.process_samples(&samples).unwrap();
        assert_eq!(
            features.mel_frames as u64,
            trace["mel_frames"].as_u64().unwrap()
        );
        assert_eq!(features.values.len() * 4, mel.len());
        let diff = features
            .values
            .iter()
            .zip(mel.as_chunks::<4>().0.iter())
            .map(|(&a, b)| (a - f32::from_le_bytes(*b)).abs())
            .fold(0.0f32, f32::max);
        assert!(diff <= 1e-3, "{id}: mel maximum absolute difference {diff}");
    }
}

#[test]
fn qwen_language_table_matches_pinned_config() {
    let config = json(parity().join("checkpoints/qwen3-asr-1.7b/config.json"));
    let names = config["thinker_config"]["audio_config"]["support_languages"]
        .as_array()
        .or_else(|| config["support_languages"].as_array())
        .or_else(|| config["thinker_config"]["support_languages"].as_array())
        .unwrap();
    assert_eq!(names.len(), qwen::LANGUAGES.len());
    let mut codes = HashSet::new();
    for &(code, name) in qwen::LANGUAGES {
        assert!(codes.insert(code));
        assert!(names.iter().any(|v| v == name));
        assert_eq!(qwen::language_name(code).unwrap(), name);
        assert_eq!(qwen::language_code(name), Some(code));
    }
    assert_eq!(qwen::language_name("en-GB").unwrap(), "English");
    assert_eq!(qwen::language_name("EN-us").unwrap(), "English");
    assert!(matches!(
        qwen::language_name("zz-ZZ"),
        Err(synapse_speech::Error::UnsupportedLanguage(_))
    ));
    assert_eq!(qwen::language_code("None"), None);
    assert_eq!(qwen::language_code("Elvish"), None);
}

#[cfg(not(feature = "qwen"))]
#[test]
fn qwen_feature_off_keeps_parakeet_dependency_graph() {
    let output = std::process::Command::new(env!("CARGO"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "tree",
            "--locked",
            "--offline",
            "-p",
            "synapse-speech",
            "--no-default-features",
            "--edges",
            "normal",
            "--prefix",
            "none",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tree = String::from_utf8(output.stdout).unwrap();
    for absent in [
        "tokenizers ",
        "onig ",
        "regex ",
        "synapse-module ",
        "synapse-daemon ",
        "synapse-store ",
    ] {
        assert!(
            !tree.contains(absent),
            "unexpected dependency: {absent}\n{tree}"
        );
    }
    for present in ["rustfft ", "serde ", "serde_json ", "sha2 ", "thiserror "] {
        assert!(
            tree.contains(present),
            "lost Parakeet dependency: {present}"
        );
    }
}

#[cfg(feature = "qwen")]
#[test]
fn qwen_prompt_boundaries_match_official_tokenizer() {
    use qwen::{PromptRequest, Tokenizer};
    let fixtures: Value =
        serde_json::from_slice(include_bytes!("fixtures/qwen-prompts.json")).unwrap();
    let tokenizer = Tokenizer::pinned();
    assert_eq!(fixtures["pretokenizer_sha256"], qwen::PRETOKENIZER_SHA256);
    assert_eq!(
        tokenizer.encode("e\u{301}").unwrap(),
        tokenizer.encode("é").unwrap()
    );
    for row in fixtures["cases"].as_array().unwrap() {
        let terms: Vec<_> = row["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let rendered = tokenizer
            .render(PromptRequest {
                terms: &terms,
                prompt: row["prompt"].as_str(),
                language: row["language"].as_str(),
                audio_tokens: row["audio_tokens"].as_u64().unwrap() as usize,
                bias_limit: row["limit"].as_u64().unwrap() as usize,
            })
            .unwrap();
        assert_eq!(
            serde_json::to_value(&rendered.ids).unwrap(),
            row["prompt_ids"]
        );
        assert_eq!(serde_json::to_value(rendered.span).unwrap(), row["span"]);
        assert_eq!(
            serde_json::to_value(rendered.bias_applied).unwrap(),
            row["bias_applied"]
        );
        assert_eq!(rendered.context, row["context"].as_str().unwrap());
        assert_eq!(
            rendered.context_tokens,
            row["context_tokens"].as_u64().unwrap() as usize
        );
        assert!(
            rendered.ids[rendered.span.start..rendered.span.start + rendered.span.len]
                .iter()
                .all(|&id| id == qwen::AUDIO_PAD)
        );
        assert_eq!(
            rendered
                .ids
                .iter()
                .filter(|&&id| id == qwen::AUDIO_PAD)
                .count(),
            rendered.span.len
        );
    }
    for text in ["<|audio_pad|>", "<|im_start|>"] {
        let ids = tokenizer.encode(text).unwrap();
        assert!(!ids
            .iter()
            .any(|id| [qwen::AUDIO_PAD, 151644, qwen::ASR_TEXT].contains(id)));
    }
    assert_eq!(
        tokenizer.encode("<asr_text>").unwrap(),
        vec![qwen::ASR_TEXT]
    );
    let terms = ["Ada", "CortexKit"];
    let cost = tokenizer.context_tokens(&terms, None).unwrap();
    for (limit, used) in [(cost, 2), (cost - 1, 1)] {
        let result = tokenizer
            .render(PromptRequest {
                terms: &terms,
                prompt: None,
                language: None,
                audio_tokens: 1,
                bias_limit: limit,
            })
            .unwrap();
        assert_eq!(result.bias_applied.terms_used, used);
        assert!(result.context_tokens <= limit);
    }
    assert!(Tokenizer::from_pinned_bytes(b"{}", b"#version: 0.2\n").is_err());
    assert!(tokenizer
        .render(PromptRequest {
            terms: &[],
            prompt: None,
            language: Some("zz"),
            audio_tokens: 1,
            bias_limit: 0
        })
        .is_err());
}

#[test]
fn qwen_every_sealed_frame_count_and_language_is_supported() {
    let corpus = corpus();
    let mut checked = 0;
    let mut languages = HashSet::new();
    for case in corpus["cases"].as_array().unwrap() {
        let trace = trace(case["id"].as_str().unwrap());
        if trace["mel_frames"].is_null() {
            continue;
        }
        let frames = qwen::mel_frames(trace["sample_count"].as_u64().unwrap());
        assert_eq!(frames, trace["mel_frames"].as_u64().unwrap());
        assert_eq!(frames, trace["valid_mel_frames"].as_u64().unwrap());
        assert_eq!(
            qwen::audio_tokens(frames),
            trace["span"]["len"].as_u64().unwrap()
        );
        if let Some(name) = trace["language"].as_str() {
            let code = qwen::language_code(name)
                .unwrap_or_else(|| panic!("unknown sealed language {name}"));
            assert_eq!(qwen::language_name(code).unwrap(), name);
            languages.insert(name.to_owned());
        }
        checked += 1;
    }
    assert_eq!(checked, 6559);
    assert!(languages.len() >= 8);
}

#[cfg(feature = "qwen")]
#[test]
fn qwen_every_sealed_prompt_and_span_matches_reference() {
    let corpus = corpus();
    let tokenizer = qwen::Tokenizer::pinned();
    let mut checked = 0;
    for case in corpus["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let trace = trace(id);
        if trace["prompt_ids"].is_null() {
            continue;
        }
        let terms: Vec<_> = corpus["term_lists"][case["term_lists"]["terms"].as_str().unwrap()]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let rendered = tokenizer
            .render(qwen::PromptRequest {
                terms: &terms,
                prompt: case["prompt"].as_str(),
                language: case["language_hint"].as_str(),
                audio_tokens: qwen::audio_tokens(trace["valid_mel_frames"].as_u64().unwrap())
                    as usize,
                bias_limit: usize::MAX,
            })
            .unwrap();
        assert_eq!(
            serde_json::to_value(&rendered.ids).unwrap(),
            trace["prompt_ids"],
            "{id}"
        );
        assert_eq!(
            serde_json::to_value(rendered.span).unwrap(),
            trace["span"],
            "{id}"
        );
        assert_eq!(
            rendered.context_tokens,
            trace["context_tokens"].as_u64().unwrap() as usize,
            "{id}"
        );
        checked += 1;
    }
    assert_eq!(checked, 6559);
}

#[cfg(feature = "qwen")]
#[test]
fn qwen_every_sealed_generated_output_matches_reference() {
    let tokenizer = qwen::Tokenizer::pinned();
    let mut checked = 0;
    for case in corpus()["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let trace = trace(id);
        if trace["generated_ids"].is_null() {
            continue;
        }
        let ids: Vec<u32> = serde_json::from_value(trace["generated_ids"].clone()).unwrap();
        let hint = case["language_hint"]
            .as_str()
            .map(|name| qwen::language_code(name).unwrap());
        let parsed = tokenizer
            .parse(&ids, hint)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(parsed.text, trace["text"].as_str().unwrap(), "{id}");
        assert_eq!(
            parsed.language.as_deref(),
            trace["language"].as_str().and_then(qwen::language_code),
            "{id}"
        );
        checked += 1;
    }
    assert_eq!(checked, 6559);
}

#[cfg(feature = "qwen")]
#[test]
fn qwen_sealed_generated_parser_boundaries() {
    use synapse_speech::Error;
    let bytes = include_bytes!("fixtures/qwen-generated.json");
    assert_eq!(bytes.len(), 1294);
    assert_eq!(
        sha(bytes),
        "208d5b49e6f416f531a208ac02ca054bc3e28f13bcce9609bee2c27d7d0f476f"
    );
    let fixtures: Value = serde_json::from_slice(bytes).unwrap();
    let tokenizer = qwen::Tokenizer::pinned();
    for case in fixtures["cases"].as_array().unwrap() {
        let ids: Vec<u32> = serde_json::from_value(case["generated_ids"].clone()).unwrap();
        let result = tokenizer.parse(&ids, case["hint"].as_str());
        if let Some(message) = case["error"].as_str() {
            assert!(
                matches!(result, Err(Error::InvalidAsrOutput(actual)) if actual == message),
                "{}",
                case["id"]
            );
        } else {
            let parsed = result.unwrap();
            assert_eq!(
                parsed.text,
                case["text"].as_str().unwrap(),
                "{}",
                case["id"]
            );
            assert_eq!(
                parsed.language.as_deref(),
                case["language"].as_str(),
                "{}",
                case["id"]
            );
        }
    }
    for &(code, _) in qwen::LANGUAGES {
        let parsed = tokenizer
            .parse(&[58929, 558, 13, 151645], Some(code))
            .unwrap();
        assert_eq!(parsed.text, "Venice.");
        assert_eq!(parsed.language.as_deref(), Some(code));
    }
    assert_eq!(
        tokenizer
            .parse(&[], Some("EN-us"))
            .unwrap()
            .language
            .as_deref(),
        Some("en")
    );
    assert!(
        matches!(tokenizer.parse(&[], Some("zz")), Err(Error::UnsupportedLanguage(code)) if code == "zz")
    );
    assert!(matches!(
        tokenizer.parse(&[u32::MAX], Some("en")),
        Err(Error::InvalidAsrOutput(_))
    ));
    let prompt = tokenizer
        .render(qwen::PromptRequest {
            terms: &["<asr_text>"],
            prompt: None,
            language: None,
            audio_tokens: 1,
            bias_limit: 100,
        })
        .unwrap();
    assert!(prompt.ids.contains(&qwen::ASR_TEXT));
    assert!(matches!(
        tokenizer.parse(&[58929, 558, 13], None),
        Err(Error::InvalidAsrOutput("missing <asr_text> marker"))
    ));
}

#[test]
fn word_normalizer_matches_both_sealed_corpora() {
    use synapse_parity::speech::{normalize_words as reference_normalize, NORMALIZER};
    use synapse_speech::{normalize_words, WORD_NORMALIZER_REVISION};
    assert_eq!(WORD_NORMALIZER_REVISION, NORMALIZER);
    let mut checked = 0;
    let mut check = |case: &Value| {
        for key in ["ground_truth", "text"] {
            if let Some(text) = case[key].as_str() {
                assert_eq!(
                    normalize_words(text),
                    reference_normalize(text),
                    "{key}: {text}"
                );
                checked += 1;
            }
        }
    };
    for case in corpus()["cases"].as_array().unwrap() {
        let trace = trace(case["id"].as_str().unwrap());
        check(&trace);
    }
    let index = json(parity().join("fixtures/index.json"));
    let seal = &index["parakeet-tdt-0.6b-v3.ref-v1.transformers-5.16.1.seed-0"];
    let bytes = fs::read(parity().join(seal["path"].as_str().unwrap())).unwrap();
    assert_eq!(bytes.len() as u64, seal["bytes"].as_u64().unwrap());
    assert_eq!(sha(&bytes), seal["sha256"].as_str().unwrap());
    let parakeet: Value = serde_json::from_slice(&bytes).unwrap();
    for case in parakeet["cases"].as_array().unwrap() {
        check(case);
    }
    // Parakeet's 47 synthetic frontend-boundary cases have no human reference;
    // Qwen's over-limit refusal has ground truth but no generated transcript.
    assert_eq!(checked, 2 * 6559 + 1 + 1200 + 1247);
}

/// Public PCM is reconstructed with the pinned corpus recipes and checked against
/// each trace's model-input digest. Set SYNAPSE_QWEN_PARITY_INPUTS to that mirror.
#[test]
#[ignore = "requires the public PCM and sealed mel tensor mirror"]
fn qwen_all_mel_reference() {
    let root = PathBuf::from(
        std::env::var_os("SYNAPSE_QWEN_PARITY_INPUTS").expect("public PCM/mel mirror required"),
    );
    let corpus = corpus();
    let frontend = qwen::Frontend::new();
    let mut seen: HashMap<String, (String, usize, usize)> = HashMap::new();
    let mut cases = 0;
    let mut largest = 0.0f32;
    let mut worst = String::new();
    for case in corpus["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let trace = trace(id);
        if trace["mel"].is_null() {
            continue;
        }
        let recording = case["recording"].as_str().unwrap();
        let seal = &trace["mel"];
        let digest = seal["sha256"].as_str().unwrap();
        let frames = trace["mel_frames"].as_u64().unwrap() as usize;
        let valid = trace["valid_mel_frames"].as_u64().unwrap() as usize;
        if let Some(prior) = seen.get(recording) {
            assert_eq!(
                prior,
                &(digest.to_owned(), frames, valid),
                "same PCM must have identical mel: {id}"
            );
        } else {
            let raw = fs::read(root.join(format!("{recording}.pcm"))).unwrap();
            assert_eq!(
                sha(&raw),
                trace["model_input_audio_sha256"].as_str().unwrap(),
                "{id}"
            );
            assert_eq!(
                raw.len(),
                trace["sample_count"].as_u64().unwrap() as usize * 4,
                "{id}"
            );
            let samples: Vec<_> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            let reference = fs::read(root.join(format!("{digest}.mel"))).unwrap();
            assert_eq!(
                reference.len(),
                seal["bytes"].as_u64().unwrap() as usize,
                "{id}"
            );
            assert_eq!(sha(&reference), digest, "{id}");
            let features = frontend.process_samples(&samples).unwrap();
            assert_eq!(features.mel_frames, frames, "{id}");
            assert_eq!(features.valid_mel_frames, valid, "{id}");
            assert_eq!(
                features.audio_tokens,
                trace["span"]["len"].as_u64().unwrap() as usize,
                "{id}"
            );
            assert_eq!(features.values.len() * 4, reference.len(), "{id}");
            let diff = features
                .values
                .iter()
                .zip(reference.as_chunks::<4>().0.iter())
                .map(|(&a, b)| (a - f32::from_le_bytes(*b)).abs())
                .fold(0.0f32, f32::max);
            assert!(diff <= 1e-3, "{id}: mel max absolute difference {diff}");
            if diff > largest {
                largest = diff;
                worst = id.to_owned();
            }
            seen.insert(recording.to_owned(), (digest.to_owned(), frames, valid));
        }
        cases += 1;
    }
    assert_eq!(cases, 6559);
    assert_eq!(seen.len(), 1556);
    println!(
        "{cases} fixture cases, {} unique PCM tensors; maximum mel difference {largest} at {worst}",
        seen.len()
    );
}
