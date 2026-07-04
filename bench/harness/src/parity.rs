//! Shared parity math and bench I/O. All lanes MUST use these so parity
//! numbers are computed identically across runtimes (a lane-local cosine
//! variant would silently break cross-lane comparability).

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// One corpus chunk as consumed by every lane.
#[derive(Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
}

/// One workload-B prompt as consumed by every lane.
#[derive(Deserialize)]
pub struct Prompt {
    pub id: String,
    pub prompt: String,
}

pub fn load_corpus(path: &Path, limit: Option<usize>) -> Result<Vec<Chunk>> {
    let mut chunks = load_jsonl::<Chunk>(path)?;
    if let Some(limit) = limit {
        chunks.truncate(limit);
    }
    anyhow::ensure!(!chunks.is_empty(), "empty corpus: {}", path.display());
    Ok(chunks)
}

pub fn load_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    std::io::BufReader::new(file)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}

/// Reference vectors: JSONL of {id, vec}.
pub fn load_reference(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    #[derive(Deserialize)]
    struct Row {
        id: String,
        vec: Vec<f32>,
    }
    Ok(load_jsonl::<Row>(path)?.into_iter().map(|r| (r.id, r.vec)).collect())
}

/// Cosine similarity in f64 accumulation (bf16/f16 lanes need the headroom).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-12)
}

/// Mean cosine of produced vectors against a reference over intersecting ids.
/// Returns (mean, matched_count); None mean when nothing intersects.
pub fn mean_parity(
    produced: impl IntoIterator<Item = (String, Vec<f32>)>,
    reference: &HashMap<String, Vec<f32>>,
) -> (Option<f64>, usize) {
    let mut sum = 0f64;
    let mut n = 0usize;
    for (id, vec) in produced {
        if let Some(ref_vec) = reference.get(&id) {
            sum += cosine(&vec, ref_vec);
            n += 1;
        }
    }
    ((n > 0).then(|| sum / n as f64), n)
}
