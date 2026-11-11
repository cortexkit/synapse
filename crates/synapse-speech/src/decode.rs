use serde::{Deserialize, Serialize};

use crate::{Error, Result, BLANK, DURATIONS, MAX_SYMBOLS_PER_FRAME, VOCAB_SIZE};

pub type Triple = [u32; 3];

/// Engine-owned neural state. `reset` must zero the LSTM and run the initial
/// blank predictor. Only `accept` may advance the predictor; a blank preserves
/// its state and output. Encoder projections remain inside the engine.
pub trait Joint {
    fn reset(&mut self) -> Result<()>;
    fn evaluate(&mut self, frame_index: usize) -> Result<JointLogits<'_>>;
    fn accept(&mut self, token: u32) -> Result<()>;
}

pub struct JointLogits<'a> {
    pub tokens: &'a [f32],
    pub durations: &'a [f32],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceStep {
    pub frame_index: u32,
    pub token_top1: u32,
    pub token_top2: u32,
    pub duration_top1: u32,
    pub duration_top2: u32,
    pub token_logit1: f32,
    pub token_logit2: f32,
    pub duration_logit1: f32,
    pub duration_logit2: f32,
    pub emitted: bool,
    pub cap_hit: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeOutput {
    pub triples: Vec<Triple>,
    /// Empty when observation was not requested. Observation never changes the
    /// number, order or inputs of neural evaluations.
    pub trace: Vec<TraceStep>,
}

fn top_two(logits: &[f32]) -> Result<(u32, u32, f32, f32)> {
    if logits.len() < 2 || logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidJoint("head requires finite logits"));
    }
    let (mut first, mut second) = if logits[0] >= logits[1] {
        (0, 1)
    } else {
        (1, 0)
    };
    for i in 2..logits.len() {
        if logits[i] > logits[first] {
            second = first;
            first = i;
        } else if logits[i] > logits[second] {
            second = i;
        }
    }
    Ok((first as u32, second as u32, logits[first], logits[second]))
}

/// Bounded TDT greedy decoding. Each visited encoder frame has at most K joint
/// evaluations; only a zero advance at K forces an extra frame of progress.
/// Progress reports completed encoder frames, every 125 frames (10 seconds).
pub fn greedy<J: Joint>(
    joint: &mut J,
    valid_frames: usize,
    observe: bool,
    mut progress: impl FnMut(usize),
) -> Result<DecodeOutput> {
    if valid_frames > u32::MAX as usize {
        return Err(Error::InvalidJoint(
            "encoder length exceeds trace index range",
        ));
    }
    joint.reset()?;
    let mut output = DecodeOutput {
        triples: Vec::new(),
        trace: Vec::new(),
    };
    let (mut frame, mut evaluations, mut next_progress) = (0usize, 0usize, 125usize);
    while frame < valid_frames {
        let logits = joint.evaluate(frame)?;
        if logits.tokens.len() != VOCAB_SIZE || logits.durations.len() != DURATIONS.len() {
            return Err(Error::InvalidJoint("head shape mismatch"));
        }
        let (tok1, tok2, tl1, tl2) = top_two(logits.tokens)?;
        let (dur1, dur2, dl1, dl2) = top_two(logits.durations)?;
        let duration = DURATIONS[dur1 as usize];
        let emitted = tok1 != BLANK;
        evaluations += 1;
        let advance = if emitted { duration } else { duration.max(1) };
        let cap_hit = evaluations == MAX_SYMBOLS_PER_FRAME && advance == 0;
        if observe {
            output.trace.push(TraceStep {
                frame_index: frame as u32,
                token_top1: tok1,
                token_top2: tok2,
                duration_top1: dur1,
                duration_top2: dur2,
                token_logit1: tl1,
                token_logit2: tl2,
                duration_logit1: dl1,
                duration_logit2: dl2,
                emitted,
                cap_hit,
            });
        }
        if emitted {
            output.triples.push([tok1, frame as u32, duration]);
            joint.accept(tok1)?;
        }
        if advance > 0 || cap_hit {
            frame = frame.saturating_add(advance.max(1) as usize);
            evaluations = 0;
            while next_progress <= frame.min(valid_frames) {
                progress(next_progress);
                next_progress += 125;
            }
        }
    }
    Ok(output)
}
