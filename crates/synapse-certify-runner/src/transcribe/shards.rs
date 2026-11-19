//! Preflight the objects charged to the media store, in fixture order.
use super::{err, refuse, text, SpeechFixture};
use serde_json::Value;
use std::{ops::Range, path::PathBuf};
use synapse_certify::Result;
use synapse_parity::speech;

pub(super) const SHARD_BYTES: u64 = 768 * 1024 * 1024;
const INLINE_RESULT_BYTES: u64 = 3 * 1024 * 1024;

pub(super) struct Input<'a> {
    pub fixture: &'a SpeechFixture,
    pub path: PathBuf,
    pub digest: String,
    pub audio_bytes: u64,
    pub result_bytes: u64,
    pub rerun_bytes: u64,
}

pub(super) fn inputs<'a>(
    root: &std::path::Path,
    fixtures: &'a [SpeechFixture],
    max_duration_ms: u64,
) -> Result<Vec<Input<'a>>> {
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(root.join("inputs.json")).map_err(err)?)
            .map_err(err)?;
    fixtures
        .iter()
        .map(|fixture| {
            let entry = &manifest[&fixture.id];
            let file = std::path::Path::new(text(entry, "path")?);
            if file
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err(refuse("invalid audio cache path"));
            }
            let path = root.join(file);
            let audio_bytes = std::fs::metadata(&path).map_err(err)?.len();
            let in_limit = fixture.duration_ms <= max_duration_ms;
            Ok(Input {
                fixture,
                path,
                digest: text(entry, "sha256")?.into(),
                audio_bytes,
                result_bytes: if in_limit { result_bytes(fixture)? } else { 0 },
                rerun_bytes: if in_limit && fixture.encoder.is_some() {
                    audio_bytes
                } else {
                    0
                },
            })
        })
        .collect()
}

fn result_bytes(fixture: &SpeechFixture) -> Result<u64> {
    // Encoder and trace bytes live in observation pages, not media objects.
    // Reserve the worst-case JSON size: each emission can add a token to both
    // text and a timed word (at most 32 escaped bytes each for the Parakeet v3
    // vocabulary sealed by the fixture set),
    // one word's JSON framing (at most 46 bytes), and 16 base64 triple bytes.
    // The fixed allowance covers the result envelope and observation descriptors.
    let bound = fixture
        .valid_encoder_frames
        .checked_mul(speech::CAP as u64 * 128)
        .and_then(|bytes| bytes.checked_add(1024))
        .ok_or_else(|| refuse("speech result size overflow"))?;
    Ok(if bound > INLINE_RESULT_BYTES {
        bound
    } else {
        0
    })
}

pub(super) fn partition(inputs: &[Input<'_>]) -> Result<Vec<Range<usize>>> {
    let mut shards = Vec::new();
    let mut start = 0;
    let mut stored: u64 = 0;
    for (index, input) in inputs.iter().enumerate() {
        let retained = input.audio_bytes.saturating_add(input.result_bytes);
        // The candidate requires a separate upload to repeat inference with
        // observation disabled. It deletes that upload when the rerun ends.
        let peak = retained.saturating_add(input.rerun_bytes);
        if peak >= SHARD_BYTES {
            return Err(refuse(format!(
                "speech fixture {} needs {peak} media bytes; shard limit is {SHARD_BYTES}",
                input.fixture.id
            )));
        }
        if stored.saturating_add(peak) >= SHARD_BYTES {
            shards.push(start..index);
            start = index;
            stored = 0;
        }
        stored += retained;
    }
    if start < inputs.len() {
        shards.push(start..inputs.len());
    }
    Ok(shards)
}
