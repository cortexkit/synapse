//! Compiles a JSON Schema grammar into a wire-serializable
//! `TokenIdJsonConstraint` for distribution to decode workers.
//!
//! The grammar compiler and token vocabulary live in the owned Metal decode
//! stack, whose crates are macOS-only dependencies of this package. On other
//! platforms this binary still exists (Cargo builds every discovered bin on
//! every platform) but refuses at startup instead of failing to compile.

#[cfg(target_os = "macos")]
mod real {
    use std::path::PathBuf;

    use anyhow::Result;
    use clap::Parser;
    use owned_decode_worker::protocol::TokenIdJsonConstraint;
    use sha2::{Digest, Sha256};
    use synapse_core::Fingerprint;
    use synapse_engine_owned::owned_decode_engine::TokenVocabulary;
    use synapse_module::owned_decode_grammar_scheduler::{
        compile_grammar, CompileContext, GrammarSubsetManifest,
    };
    use tokenizers::Tokenizer;

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        decode_fingerprint: String,
        #[arg(long)]
        grammar: String,
    }

    pub fn main() -> Result<()> {
        let args = Args::parse();
        let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|error| {
            anyhow::anyhow!("load tokenizer {}: {error}", args.tokenizer.display())
        })?;
        let vocabulary = TokenVocabulary::from_tokenizer(&tokenizer)?;
        let vocabulary_digest = token_vocabulary_digest(&vocabulary);
        let compiled = compile_grammar(
            &args.grammar,
            &CompileContext {
                base_decode_fingerprint: Fingerprint(args.decode_fingerprint),
                tokenizer_vocabulary_digest: vocabulary_digest,
            },
            &GrammarSubsetManifest::default(),
        )
        .map_err(|error| anyhow::anyhow!("compile JSON constraint: {}", error.message))?
        .constraint;
        let runtime = compiled.constraint_runtime_identity;
        let wire = TokenIdJsonConstraint {
            encoding_id: compiled.representation_revision,
            constraint_runtime_identity: runtime.digest(),
            constraint_fingerprint: compiled.constraint_fingerprint.0,
            grammar_subset_revision: runtime.grammar_subset_revision,
            grammar_compiler_revision: runtime.grammar_compiler_revision,
            tokenizer_vocabulary_digest: compiled.tokenizer_vocabulary_digest,
            limits_manifest_id: compiled.limits_manifest_id,
            worker_constraint_runtime_revision: runtime.worker_constraint_runtime_revision,
            canonical_schema_digest: compiled.canonical_schema_digest,
            initial_state_encoding: compiled.initial_state_encoding,
            initial_state_digest: compiled.initial_state_digest,
            compiled_automaton_digest: compiled.compiled_automaton_digest,
            automaton_bytes: compiled.automaton_bytes,
        };
        println!("{}", serde_json::to_string(&wire)?);
        Ok(())
    }

    fn token_vocabulary_digest(vocabulary: &TokenVocabulary) -> String {
        let mut hasher = Sha256::new();
        for token_id in 0..vocabulary.len() {
            if let Some(piece) = vocabulary.token_piece(token_id as u32) {
                hasher.update(piece);
            }
            hasher.update([0]);
        }
        hex::encode(hasher.finalize())
    }
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    real::main()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("compile_constraint requires macOS: the owned Metal decode stack is not built on this platform");
    std::process::exit(1);
}
