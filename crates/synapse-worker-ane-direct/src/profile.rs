//! Opt-in Apple Neural Engine timings; host evaluation includes submission and waiting.
use serde_json::{json, Value};
use std::{io::Write, path::PathBuf, time::Instant};

pub struct LaneProfile {
    output: Option<PathBuf>,
    started: Instant,
    last: Instant,
    phases: Vec<Value>,
    layers: Vec<Value>,
}

impl LaneProfile {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            output: std::env::var_os("SYNAPSE_ANE_PROFILE_DIR").map(PathBuf::from),
            started: now,
            last: now,
            phases: Vec::new(),
            layers: Vec::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.output.is_some()
    }

    pub fn phase(&mut self, name: &str) {
        if self.enabled() {
            let now = Instant::now();
            self.phases.push(
                json!({"stage": name, "ms": now.duration_since(self.last).as_secs_f64() * 1000.0}),
            );
            self.last = now;
        }
    }

    pub fn layer(
        &mut self,
        layer: usize,
        prepare: std::time::Duration,
        evaluate: std::time::Duration,
        created: bool,
    ) {
        self.layers.push(json!({"layer": layer, "prepare_ms": prepare.as_secs_f64() * 1000.0, "submit_wait_ms": evaluate.as_secs_f64() * 1000.0, "request_created": created}));
        self.last = Instant::now();
    }

    pub fn hardware_layer(&mut self, layer: usize, wall: std::time::Duration, hw_ns: u64) {
        self.layers.push(json!({"layer":layer, "wall_ms":wall.as_secs_f64()*1000.0, "hardware_ms":hw_ns as f64/1_000_000.0}));
        self.last = Instant::now();
    }

    pub fn finish(self, mut record: Value) {
        if let Some(output) = self.output {
            record["wall_ms"] = json!(self.started.elapsed().as_secs_f64() * 1000.0);
            record["unix_us"] = json!(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros());
            record["phases"] = json!(self.phases);
            record["layers"] = json!(self.layers);
            // The module limits forwarded worker stderr lines. Write evidence
            // separately to preserve every row without changing the wire protocol.
            // Timing ends before evidence I/O.
            let path = output.join(format!("worker-{}.jsonl", std::process::id()));
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = file.write_all(format!("{record}\n").as_bytes());
            }
        }
    }
}
