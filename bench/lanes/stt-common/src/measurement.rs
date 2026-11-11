use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Transcript {
    pub id: String,
    pub text: Option<String>,
    pub seconds: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerResult {
    pub engine: String,
    pub version: String,
    pub commit: String,
    pub backend: String,
    pub cold_start_seconds: f64,
    pub warm_wall_seconds: f64,
    pub peak_rss_bytes: u64,
    pub gpu_peak_bytes: Option<u64>,
    pub energy_joules: Option<f64>,
    pub transcripts: Vec<Transcript>,
}

pub fn validate_measurements(result: &WorkerResult) -> Result<()> {
    ensure!(
        !result.engine.is_empty() && !result.version.is_empty() && !result.commit.is_empty(),
        "missing engine provenance"
    );
    ensure!(
        result.cold_start_seconds.is_finite() && result.cold_start_seconds >= 0.0,
        "invalid cold start"
    );
    ensure!(
        result.warm_wall_seconds.is_finite() && result.warm_wall_seconds > 0.0,
        "invalid warm wall time"
    );
    ensure!(
        result.peak_rss_bytes > 0,
        "missing resident memory measurement"
    );
    if let Some(energy) = result.energy_joules {
        ensure!(
            energy.is_finite() && energy >= 0.0,
            "invalid energy measurement"
        );
    }
    for transcript in &result.transcripts {
        match (&transcript.text, transcript.seconds, &transcript.error) {
            (Some(_), Some(seconds), None) => ensure!(
                seconds.is_finite() && seconds > 0.0,
                "invalid utterance latency"
            ),
            (None, None, Some(error)) => ensure!(!error.is_empty(), "empty utterance error"),
            _ => anyhow::bail!("{}: expected text+seconds or error", transcript.id),
        }
    }
    Ok(())
}

pub fn percentile(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = (percentile * sorted.len() as f64).ceil() as usize;
    Some(sorted[index.saturating_sub(1).min(sorted.len() - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stt_measurement_percentiles_use_nearest_rank() {
        assert_eq!(percentile(&[], 0.5), None);
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.5), Some(2.0));
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.95), Some(4.0));
    }

    #[test]
    fn stt_measurement_rejects_missing_or_nonfinite_latency() {
        let mut result = WorkerResult {
            engine: "test".into(),
            version: "1".into(),
            commit: "abc".into(),
            backend: "test".into(),
            cold_start_seconds: 0.0,
            warm_wall_seconds: 1.0,
            peak_rss_bytes: 1,
            gpu_peak_bytes: None,
            energy_joules: None,
            transcripts: vec![Transcript {
                id: "one".into(),
                text: Some("hello".into()),
                seconds: Some(1.0),
                error: None,
            }],
        };
        assert!(validate_measurements(&result).is_ok());
        result.transcripts[0].seconds = Some(f64::NAN);
        assert!(validate_measurements(&result).is_err());
        result.transcripts[0].seconds = None;
        assert!(validate_measurements(&result).is_err());
        result.transcripts[0].text = None;
        result.transcripts[0].error = Some("locale unavailable".into());
        assert!(validate_measurements(&result).is_ok());
    }
}
