use std::collections::BTreeMap;

use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Precision {
    F16,
    F32,
}

impl Precision {
    pub fn parse(value: Option<&str>) -> Result<Self, &'static str> {
        match value {
            None | Some("f16") => Ok(Self::F16),
            Some("f32") => Ok(Self::F32),
            _ => Err("SYNAPSE_CUDA_SPEECH_PRECISION must be f16 or f32"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }
}

#[derive(Default)]
pub(super) struct Timings(BTreeMap<&'static str, Value>);

impl Timings {
    pub fn record(&mut self, id: &str, refused: bool, wall_s: f64, observed_request_s: f64) {
        let group = if id.starts_with("ladder-") {
            "ladders"
        } else {
            "short_clips"
        };
        let row = self.0.entry(group).or_insert_with(|| {
            json!({"fixtures": 0, "refusals": 0, "wall_elapsed_s": 0., "observed_request_s": 0.})
        });
        row["fixtures"] = json!(row["fixtures"].as_u64().unwrap() + 1);
        row["refusals"] = json!(row["refusals"].as_u64().unwrap() + u64::from(refused));
        row["wall_elapsed_s"] = json!(row["wall_elapsed_s"].as_f64().unwrap() + wall_s);
        row["observed_request_s"] =
            json!(row["observed_request_s"].as_f64().unwrap() + observed_request_s);
    }

    pub fn report(&self) -> Value {
        json!(self.0)
    }
}

#[test]
fn transcribe_measurement_precision_defaults_to_unchanged_f16_and_rejects_typos() {
    assert_eq!(Precision::parse(None), Ok(Precision::F16));
    assert_eq!(Precision::parse(Some("f16")), Ok(Precision::F16));
    assert_eq!(Precision::parse(Some("f32")), Ok(Precision::F32));
    assert_eq!(Precision::F16.label(), "f16");
    assert_eq!(Precision::F32.label(), "f32");
    assert!(Precision::parse(Some("tf32")).is_err());
    assert!(Precision::parse(Some("")).is_err());
}

#[test]
fn transcribe_measurement_timing_separates_ladders_refusals_and_short_clips() {
    let mut timings = Timings::default();
    timings.record("fleurs-de-test-a", false, 2., 1.);
    timings.record("silence-30s", false, 4., 3.);
    timings.record("ladder-1025", false, 8., 5.);
    timings.record("ladder-8193", true, 0.5, 0.25);
    assert_eq!(
        timings.report(),
        json!({"short_clips": {"fixtures": 2, "refusals": 0, "wall_elapsed_s": 6., "observed_request_s": 4.},
            "ladders": {"fixtures": 2, "refusals": 1, "wall_elapsed_s": 8.5, "observed_request_s": 5.25}})
    );
}
