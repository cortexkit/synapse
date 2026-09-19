use std::fs;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cortexkit_log::{Config, SegmentRetention};

fn log_root(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "synapse-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ))
}

#[test]
fn fleet_log_shapes_render_byte_exactly() {
    let root = log_root("log-format");
    let config = Config {
        module_id: "synapse".to_string(),
        logs_dir: root,
        // Nothing is bound process-wide, so no line carries a bracket: the
        // module is not a harness-hosted plugin and binds no session here.
        bound: Vec::new(),
        spec: Some("debug".to_string()),
        retention: SegmentRetention::default(),
        redactor: None,
        clock: Some(Arc::new(|| {
            UNIX_EPOCH + Duration::from_millis(1_788_604_863_123)
        })),
    };
    let handle = cortexkit_log::init(config).expect("fleet logger initializes");

    tracing::debug!(
        target: "perf",
        in_flight = 2_u64,
        by_model = "embed-model:2",
        tokens_per_s = format_args!("{:.3}", 12.5),
        queue_depth = 1_u64,
        waiters = 1_u64,
        worker_rss_mb = "embed-model:384",
        "activity"
    );
    tracing::info!(
        target: "perf",
        model_id = "embed-model",
        job_id = "inline-7-2",
        lane = "ane",
        tokens = 42_u64,
        wall_ms = 17_u64,
        "job done"
    );
    tracing::info!(
        target: "worker",
        worker = "ane-0",
        stream = "stderr",
        model_id = "model with space",
        "worker diagnostic"
    );
    tracing::warn!(
        target: "admission",
        model_id = "embed-model",
        job_id = "inline-7-3",
        reason = "queue_full",
        "job refused"
    );

    let actual = fs::read_to_string(handle.path()).expect("rendered fleet log");
    // The logger name on each line is derived from the tracing target above,
    // rooted at the module id: target `perf` becomes `synapse.perf`. The
    // trailing colon separates it from the message, and there is no bracket
    // because nothing is bound.
    let expected = concat!(
        "2026-09-05T10:41:03.123Z DEBUG synapse.perf: activity in_flight=2 by_model=embed-model:2 tokens_per_s=12.500 queue_depth=1 waiters=1 worker_rss_mb=embed-model:384\n",
        "2026-09-05T10:41:03.123Z INFO  synapse.perf: job done model_id=embed-model job_id=inline-7-2 lane=ane tokens=42 wall_ms=17\n",
        "2026-09-05T10:41:03.123Z INFO  synapse.worker: worker diagnostic worker=ane-0 stream=stderr model_id=\"model with space\"\n",
        "2026-09-05T10:41:03.123Z WARN  synapse.admission: job refused model_id=embed-model job_id=inline-7-3 reason=queue_full\n",
    );
    assert_eq!(actual.as_bytes(), expected.as_bytes());
}
