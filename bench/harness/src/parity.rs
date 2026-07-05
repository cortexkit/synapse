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

/// Rank-stability check between two vector spaces over the same ids.
/// Uses every `stride`-th vector as a query against all others (self excluded)
/// and reports mean top-k neighbor overlap (Jaccard-free: |A ∩ B| / k).
/// Detects quantization-induced reordering that mean cosine hides: near-tie
/// neighbors can swap order while per-vector cosine stays > 0.99.
pub fn rank_overlap(
    a: &HashMap<String, Vec<f32>>,
    b: &HashMap<String, Vec<f32>>,
    k: usize,
    stride: usize,
) -> Result<RankOverlap> {
    let ids: Vec<&String> = {
        let mut ids: Vec<&String> = a.keys().filter(|id| b.contains_key(*id)).collect();
        ids.sort(); // deterministic across runs
        ids
    };
    anyhow::ensure!(ids.len() > k + 1, "not enough shared ids ({}) for k={k}", ids.len());

    let queries: Vec<&String> = ids.iter().step_by(stride.max(1)).copied().collect();
    let mut overlap_sum = 0f64;

    for query in &queries {
        let top_a = top_k_neighbors(query, &ids, a, k);
        let top_b = top_k_neighbors(query, &ids, b, k);
        let hits = top_a.iter().filter(|id| top_b.contains(*id)).count();
        overlap_sum += hits as f64 / k as f64;
    }

    Ok(RankOverlap {
        queries: queries.len(),
        k,
        mean_topk_overlap: overlap_sum / queries.len() as f64,
    })
}

#[derive(serde::Serialize)]
pub struct RankOverlap {
    pub queries: usize,
    pub k: usize,
    pub mean_topk_overlap: f64,
}

fn top_k_neighbors(
    query: &str,
    ids: &[&String],
    space: &HashMap<String, Vec<f32>>,
    k: usize,
) -> Vec<String> {
    let qv = &space[query];
    let mut scored: Vec<(f64, &String)> = ids
        .iter()
        .filter(|id| id.as_str() != query)
        .map(|id| (cosine(&space[id.as_str()], qv), *id))
        .collect();
    scored.sort_by(|x, y| y.0.total_cmp(&x.0));
    scored.truncate(k);
    scored.into_iter().map(|(_, id)| id.clone()).collect()
}
