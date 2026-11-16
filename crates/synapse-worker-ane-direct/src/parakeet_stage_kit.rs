//! Matched identities are attached outside the model-inference timing boundary.
use super::*;

pub(super) struct Kit {
    pub run_id: String,
    writer: File,
}

impl Kit {
    pub fn open() -> Result<Option<Self>> {
        let Ok(run_id) = std::env::var("ANE_PARAKEET_STAGE_RUN_ID") else {
            return Ok(None);
        };
        ensure!(!run_id.is_empty(), "stage run id must not be empty");
        let path = env_path("ANE_PARAKEET_REPORT")?.with_extension("stages.jsonl");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Some(Self {
            run_id,
            writer: File::create(path)?,
        }))
    }

    pub fn record(
        &mut self,
        id: &str,
        rung: usize,
        frames: usize,
        seconds: f64,
        output: &Output,
    ) -> Result<()> {
        let record = Self::record_value(&self.run_id, id, rung, frames, seconds, output);
        reporting::write_case_record(&mut self.writer, &record)
    }

    fn record_value(
        run_id: &str,
        id: &str,
        rung: usize,
        frames: usize,
        seconds: f64,
        output: &Output,
    ) -> Value {
        let mut record = output.stage_timing.clone().unwrap_or_else(|| json!({}));
        record["schema"] = json!("matched-stages-v1");
        record["engine"] = json!("synapse");
        record["run_id"] = json!(run_id);
        record["clip_id"] = json!(id);
        record["invocation_id"] = json!("primary");
        record["segment_id"] = json!(0);
        record["segment_count"] = json!(1);
        record["population"] = json!("corpus");
        record["observation"] = json!("on");
        record["warm"] = json!(true);
        record["instrumented"] = json!(output.stage_timing.is_some());
        record["model_total_ms"] = json!(seconds * 1000.0);
        record["valid_frames"] = json!(frames);
        record["padded_frames"] = json!(rung);
        record["rung"] = json!(rung);
        record["profile"] = json!(PROFILE);
        record["accepted_tokens"] = json!(output.decoded.triples.len());
        record["compiled_program_count"] = json!(24);
        record["encoder_calls"] = json!(1);
        record["encoder_program_calls"] = json!(24);
        record
    }

    pub fn warmup(
        &mut self,
        id: &str,
        rung: usize,
        frames: usize,
        seconds: f64,
        output: &Output,
    ) -> Result<()> {
        let mut record = Self::record_value(&self.run_id, id, rung, frames, seconds, output);
        record["population"] = json!("warmup");
        record["invocation_id"] = json!("warmup");
        record["warm"] = json!(false);
        reporting::write_case_record(&mut self.writer, &record)
    }
}

#[test]
fn matched_record_keeps_identity_counts_and_uninstrumented_absence() {
    let mut output = Output {
        frames: 5,
        encoder: vec![],
        decoded: synapse_speech::DecodeOutput {
            triples: vec![[7, 0, 2]],
            trace: vec![],
        },
        stage_timing: Some(
            json!({"encoder_ms": 3.0, "predictor_ms": 2.0, "decode_steps": 9, "predictor_calls": 2}),
        ),
    };
    let record = Kit::record_value("run-A", "clip-Z", 128, 5, 0.011, &output);
    assert_eq!(record["run_id"], "run-A");
    assert_eq!(record["clip_id"], "clip-Z");
    assert_eq!(record["model_total_ms"], 11.0);
    assert_eq!(record["encoder_ms"], 3.0);
    assert_eq!(record["decode_steps"], 9);
    assert_eq!(record["accepted_tokens"], 1);
    assert_eq!(record["padded_frames"], 128);
    assert_eq!(record["valid_frames"], 5);
    output.stage_timing = None;
    let off = Kit::record_value("run-A", "clip-Z", 128, 5, 0.010, &output);
    assert_eq!(off["instrumented"], false);
    assert!(off.get("encoder_ms").is_none());
}
