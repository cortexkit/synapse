use std::fs;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cortexkit_log::{Config, Lane, Retention};
use synapse_module::LOG_TAGS;

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
    cortexkit_log::declared_tags(LOG_TAGS);
    let config = Config {
        module_id: "synapse".to_string(),
        logs_dir: root,
        lane: Lane::Module,
        spec: Some("debug".to_string()),
        retention: Retention::default(),
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
    let expected = concat!(
        "2026-09-05T10:41:03.123Z DEBUG synapse tag=perf activity in_flight=2 by_model=embed-model:2 tokens_per_s=12.500 queue_depth=1 waiters=1 worker_rss_mb=embed-model:384\n",
        "2026-09-05T10:41:03.123Z INFO  synapse tag=perf job done model_id=embed-model job_id=inline-7-2 lane=ane tokens=42 wall_ms=17\n",
        "2026-09-05T10:41:03.123Z INFO  synapse tag=worker worker diagnostic worker=ane-0 stream=stderr model_id=\"model with space\"\n",
        "2026-09-05T10:41:03.123Z WARN  synapse tag=admission job refused model_id=embed-model job_id=inline-7-3 reason=queue_full\n",
    );
    assert_eq!(actual.as_bytes(), expected.as_bytes());
}
