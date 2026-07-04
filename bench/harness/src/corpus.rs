//! Corpus builder: walks a source tree, cuts files into token-budgeted chunks.
//!
//! This is a deliberately simple line-based chunker (not AFT's semantic
//! chunker). It exists so candidate-runtime integration work can start before
//! AFT exports a real chunk corpus; final published numbers should use AFT's
//! corpus. Chunk shape matches what lanes consume: {id, path, text, tokens}.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use tokenizers::Tokenizer;
use walkdir::WalkDir;

#[derive(Serialize)]
pub struct Chunk {
    pub id: String,
    pub path: String,
    pub text: String,
    pub tokens: usize,
}

const SOURCE_EXTS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "py", "go", "java", "kt", "swift", "c", "h", "cc", "cpp",
    "hpp", "cs", "rb", "php", "scala", "sh", "sql", "toml", "yaml", "yml", "md",
];

/// Max lines fed to the tokenizer per probe; keeps chunking O(n) in file size.
const MAX_CHUNK_LINES: usize = 80;

pub fn build(
    root: &Path,
    out: &Path,
    tokenizer_path: &Path,
    target: usize,
    token_budget: usize,
) -> Result<()> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(std::fs::File::create(out)?);

    let mut files: Vec<PathBuf> = WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(name == ".git" || name == "node_modules" || name == "target" || name == "dist")
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| SOURCE_EXTS.contains(&x))
                .unwrap_or(false)
        })
        .map(|e| e.into_path())
        .collect();
    files.sort(); // deterministic corpus across runs

    let mut count = 0usize;
    let mut total_tokens = 0usize;

    'outer: for file in files {
        let Ok(content) = std::fs::read_to_string(&file) else { continue };
        let rel = file.strip_prefix(root).unwrap_or(&file).to_string_lossy().to_string();
        let lines: Vec<&str> = content.lines().collect();

        let mut start = 0usize;
        while start < lines.len() {
            let end = (start + MAX_CHUNK_LINES).min(lines.len());
            let mut cut = end;
            let mut text = lines[start..cut].join("\n");
            let mut n_tokens = count_tokens(&tokenizer, &text)?;
            // Shrink until under budget.
            while n_tokens > token_budget && cut > start + 4 {
                cut = start + (cut - start) * 3 / 4;
                text = lines[start..cut].join("\n");
                n_tokens = count_tokens(&tokenizer, &text)?;
            }
            if !text.trim().is_empty() && n_tokens >= 8 {
                let chunk = Chunk {
                    id: format!("{rel}#{start}"),
                    path: rel.clone(),
                    text,
                    tokens: n_tokens,
                };
                serde_json::to_writer(&mut writer, &chunk)?;
                writer.write_all(b"\n")?;
                count += 1;
                total_tokens += n_tokens;
                if count >= target {
                    break 'outer;
                }
            }
            start = cut;
        }
    }

    writer.flush()?;
    eprintln!(
        "corpus: {count} chunks, {total_tokens} tokens total, avg {} tok/chunk -> {}",
        total_tokens.checked_div(count).unwrap_or(0),
        out.display()
    );
    anyhow::ensure!(count > 0, "no chunks produced from {}", root.display());
    Ok(())
}

fn count_tokens(tokenizer: &Tokenizer, text: &str) -> Result<usize> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))
        .context("tokenizer encode failed")?;
    Ok(enc.get_ids().len())
}
