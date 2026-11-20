//! Test-only timestamps, phase summaries and allocation evidence for the speech encoder.
use super::{vk, Arc, Arena, Device, Duration, Instant, Result, SpeechDispatch};
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader},
    process::{Command, Stdio},
    sync::mpsc,
    thread,
};

pub(crate) const CHILD: &str = "SYNAPSE_VULKAN_SPEECH_TIMING_CHILD";
pub(super) const LARGE_BUFFER: u64 = 1 << 20;
pub(crate) const PREFIX: &str = "speech_timing ";

pub(crate) fn event(value: Value) {
    eprintln!("{PREFIX}{value}");
}

/// A separate process enforces deadlines even inside a blocked Vulkan call.
#[allow(
    clippy::zombie_processes,
    reason = "try_wait reaps the child without risking an unbounded wait on a wedged driver"
)]
pub(crate) fn supervise(test: &str, finished_phase: &str) {
    let started = Instant::now();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--nocapture", "--exact", test])
        .env(CHILD, "1")
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = BufReader::new(child.stderr.take().unwrap());
    let (send, receive) = mpsc::channel();
    thread::spawn(move || {
        for line in stderr.lines() {
            let Ok(line) = line else { break };
            if send.send((Instant::now(), line)).is_err() {
                break;
            }
        }
    });
    let rung_deadline = started + Duration::from_secs(600);
    let (mut phase, mut group) = ("startup".to_owned(), Value::Null);
    let mut phase_deadline = None;
    let mut output_open = true;
    loop {
        let now = Instant::now();
        let deadline = phase_deadline.map_or(rung_deadline, |t: Instant| t.min(rung_deadline));
        if now >= deadline {
            let killed = child.kill();
            let until = Instant::now() + Duration::from_secs(3);
            let mut exited = false;
            while Instant::now() < until {
                if child.try_wait().unwrap().is_some() {
                    exited = true;
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            event(json!({
                "event": "stop", "reason": if deadline == rung_deadline { "rung_600s" } else { "layer_120s" },
                "phase": phase, "last_group": group, "wall_s": started.elapsed().as_secs_f64(),
                "child_exited": exited, "kill_error": killed.err().map(|e| e.to_string()),
            }));
            // A timeout without confirmed child exit tells the caller not to
            // launch another fixture on a device that may still be busy.
            std::process::exit(if exited { 124 } else { 125 });
        }
        let poll = (deadline - now).min(Duration::from_millis(50));
        if !output_open {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "timing child exited with {status}, phase={phase}, last_group={group}"
                );
                assert_eq!(
                    phase, finished_phase,
                    "timing child produced no completion marker"
                );
                return;
            }
            thread::sleep(poll);
            continue;
        }
        let (at, line) = match receive.recv_timeout(poll) {
            Ok(item) => item,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                output_open = false;
                continue;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
        };
        if let Some(data) = line
            .strip_prefix(PREFIX)
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
        {
            match data["event"].as_str() {
                Some("phase_start") => {
                    phase = data["phase"].as_str().unwrap().into();
                    group = Value::Null;
                    phase_deadline =
                        (data["layer"] == true).then_some(at + Duration::from_secs(120));
                }
                Some("phase_done") => phase_deadline = None,
                Some("group_start") => group = data,
                _ => {}
            }
        }
        eprintln!("{line}");
    }
}

struct Phase {
    name: String,
    started: Instant,
    groups: usize,
    gpu_ms: Option<f64>,
    submit_wall_ms: f64,
}

pub(super) struct Timing {
    device: Arc<Device>,
    queries: Option<vk::QueryPool>,
    timestamp_ns: f64,
    phase: Option<Phase>,
}

impl Timing {
    pub(super) fn new(device: &Arc<Device>) -> Result<Self> {
        let props = unsafe {
            device
                .instance
                .raw
                .get_physical_device_properties(device.physical)
        };
        let adapter = props.device_name_as_c_str()?.to_string_lossy();
        let timestamp_ns = f64::from(props.limits.timestamp_period);
        let queries = if device.timestamp_valid_bits > 0 && timestamp_ns > 0.0 {
            Some(unsafe {
                device.raw.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )?
            })
        } else {
            None
        };
        event(json!({
            "event": "queue", "adapter": adapter, "vendor_id": props.vendor_id,
            "device_id": props.device_id, "driver_version": props.driver_version,
            "timestamp_valid_bits": device.timestamp_valid_bits,
            "timestamp_period_ns": queries.map(|_| timestamp_ns),
        }));
        Ok(Self {
            device: Arc::clone(device),
            queries,
            timestamp_ns,
            phase: None,
        })
    }

    pub(super) fn enter(&mut self, name: &str, layer: bool) {
        if let Some(phase) = self.phase.take() {
            event(json!({
                "event": "phase_done", "phase": phase.name, "groups": phase.groups,
                "wall_ms": phase.started.elapsed().as_secs_f64() * 1e3,
                "submit_wall_ms": phase.submit_wall_ms, "gpu_ms": phase.gpu_ms,
            }));
        }
        self.phase = Some(Phase {
            name: name.into(),
            started: Instant::now(),
            groups: 0,
            gpu_ms: None,
            submit_wall_ms: 0.0,
        });
        event(json!({"event": "phase_start", "phase": name, "layer": layer}));
    }

    pub(super) fn begin_group(&mut self, dispatches: &[SpeechDispatch]) -> Instant {
        let phase = self.phase.as_mut().expect("timed dispatch has no phase");
        phase.groups += 1;
        let ops: Vec<_> = dispatches
            .iter()
            .map(|d| {
                json!({
                    "op": d.params.op, "label": d.label, "rows": d.params.rows,
                    "cols": d.params.cols, "inner": d.params.inner, "seq": d.params.seq,
                    "offset": d.params.offset, "window": d.params.window,
                    "logical_workgroups": d.groups,
                    "grid": [d.groups.min(65535), d.groups.div_ceil(65535), 1],
                })
            })
            .collect();
        event(
            json!({"event": "group_start", "phase": phase.name, "group": phase.groups, "ops": ops}),
        );
        Instant::now()
    }

    pub(super) unsafe fn record_start(&self, command: vk::CommandBuffer) {
        if let Some(pool) = self.queries {
            unsafe {
                self.device.raw.cmd_reset_query_pool(command, pool, 0, 2);
                self.device.raw.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    pool,
                    0,
                );
            }
        }
    }

    pub(super) unsafe fn record_end(&self, command: vk::CommandBuffer) {
        if let Some(pool) = self.queries {
            unsafe {
                self.device.raw.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    1,
                );
            }
        }
    }

    pub(super) fn finish_group(&mut self, started: Instant, submit_wall: Duration) -> Result<()> {
        let wall = started.elapsed();
        let submit_wall_ms = submit_wall.as_secs_f64() * 1e3;
        let gpu_ms = if let Some(pool) = self.queries {
            let bits = self.device.timestamp_valid_bits;
            let wrap_ms = 2f64.powi(bits as i32) * self.timestamp_ns / 1e6;
            // CPU time bounds the timestamp interval. If a full counter cycle
            // may have elapsed, the modulo result is ambiguous.
            if submit_wall_ms >= wrap_ms {
                None
            } else {
                let mut ticks = [0u64; 2];
                unsafe {
                    self.device.raw.get_query_pool_results(
                        pool,
                        0,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64,
                    )?;
                }
                let mask = u64::MAX >> (64 - bits);
                Some((ticks[1].wrapping_sub(ticks[0]) & mask) as f64 * self.timestamp_ns / 1e6)
            }
        } else {
            None
        };
        let phase = self.phase.as_mut().expect("timed dispatch has no phase");
        phase.gpu_ms = if phase.groups == 1 {
            gpu_ms
        } else {
            phase.gpu_ms.zip(gpu_ms).map(|(a, b)| a + b)
        };
        phase.submit_wall_ms += submit_wall_ms;
        event(json!({
            "event": "group_done", "phase": phase.name, "group": phase.groups,
            "wall_ms": wall.as_secs_f64() * 1e3, "submit_wall_ms": submit_wall_ms,
            "gpu_ms": gpu_ms,
        }));
        Ok(())
    }
}

impl Drop for Timing {
    fn drop(&mut self) {
        if let Some(pool) = self.queries.filter(|_| !self.device.retirement_unknown()) {
            unsafe { self.device.raw.destroy_query_pool(pool, None) };
        }
    }
}

pub(super) fn report_arena(arena: &Arena, index: u32, props: &vk::PhysicalDeviceMemoryProperties) {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let device = &arena.device;
    let budget = unsafe {
        device
            .instance
            .raw
            .enumerate_device_extension_properties(device.physical)
    }
    .is_ok_and(|extensions| {
        extensions.iter().any(|extension| {
            extension
                .extension_name_as_c_str()
                .is_ok_and(|name| name == c"VK_EXT_memory_budget")
        })
    });
    let mut budgets = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    if budget {
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budgets);
        unsafe {
            device
                .instance
                .raw
                .get_physical_device_memory_properties2(device.physical, &mut properties);
        }
    }
    for (i, heap) in props.memory_heaps[..props.memory_heap_count as usize]
        .iter()
        .enumerate()
    {
        event(json!({
            "event": "heap", "heap_index": i, "heap_bytes": heap.size,
            "device_local": heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
            "budget_bytes": budget.then_some(budgets.heap_budget[i]),
            "usage_bytes": budget.then_some(budgets.heap_usage[i]),
        }));
    }
    report_allocation(props, "arena", "arena", arena.driver_bytes, index);
    for (name, &(offset, bytes)) in &arena.slices {
        if bytes >= LARGE_BUFFER {
            event(json!({
                "event": "arena_slice", "buffer": name, "offset": offset, "bytes": bytes,
                "memory_type": index, "heap_index": props.memory_types[index as usize].heap_index,
            }));
        }
    }
}

pub(super) fn report_allocation(
    props: &vk::PhysicalDeviceMemoryProperties,
    name: &str,
    kind: &str,
    bytes: u64,
    index: u32,
) {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let memory = props.memory_types[index as usize];
    event(json!({
        "event": "allocation", "buffer": name, "kind": kind, "bytes": bytes,
        "memory_type": index, "heap_index": memory.heap_index,
        "memory_flags": memory.property_flags.as_raw(),
        "device_local": memory.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
        "host_visible": memory.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
        "host_coherent": memory.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
        "host_cached": memory.property_flags.contains(vk::MemoryPropertyFlags::HOST_CACHED),
    }));
}

#[test]
#[ignore = "requires a software Vulkan adapter and the pinned Vulkan SDK"]
fn vulkan_speech_timing_smoke() -> Result<()> {
    use super::{arena_bytes, Params, PlannedBuffer, SpeechBatch};

    if std::env::var_os(CHILD).is_none() {
        let name = module_path!().split_once("::").unwrap().1;
        supervise(
            &format!("{name}::vulkan_speech_timing_smoke"),
            "smoke_finished",
        );
        return Ok(());
    }

    let device = Device::software_for_tests()?;
    let rows = 8;
    let plan = [
        ("large_workspace", LARGE_BUFFER / 4),
        ("dummy", 1),
        ("zero", (2 * rows - 1) * 1024),
        ("values", rows * 1024),
        ("scores", rows * 8 * rows),
        ("prob", rows * 8 * rows),
        ("out", rows * 1024),
    ]
    .map(|(name, elements)| PlannedBuffer {
        name: name.into(),
        bytes: elements * 4,
        per_sub_batch: false,
    });
    let arena = Arena::new(Arc::clone(&device), &plan, arena_bytes(&plan))?;
    arena.transfer("zero", &mut vec![0; (2 * rows as usize - 1) * 4096], true)?;
    let values = vec![2.0; rows as usize * 1024];
    arena.transfer("values", &mut synapse_core::encode_f32_frame(&values), true)?;
    let mut batch = SpeechBatch::new(Arc::clone(&device))?;
    batch.policy.select_rung(2048);
    batch.timing = Some(Timing::new(&device)?);
    batch.timing.as_mut().unwrap().enter("smoke-layer", true);
    batch.push(
        &arena,
        Params {
            op: 100,
            rows: 1,
            cols: 1,
            inner: 1,
            ..Default::default()
        },
        [
            "values", "values", "out", "dummy", "dummy", "dummy", "dummy",
        ],
        1,
    )?;
    let mut gemm = [0; 4];
    arena.transfer("out", &mut gemm, false)?;
    assert_eq!(f32::from_le_bytes(gemm), 4.0);
    let shape = Params {
        rows: rows as u32,
        seq: rows as u32,
        window: rows as u32,
        ..Default::default()
    };
    for (op, bindings, count) in [
        (
            106,
            ["zero", "dummy", "scores", "zero", "zero", "zero", "zero"],
            rows * 8 * rows,
        ),
        (
            107,
            [
                "scores", "dummy", "prob", "dummy", "dummy", "dummy", "dummy",
            ],
            rows * 8,
        ),
        (
            108,
            ["prob", "dummy", "out", "dummy", "dummy", "dummy", "values"],
            rows * 1024,
        ),
    ] {
        batch.push(&arena, Params { op, ..shape }, bindings, count as u32)?;
    }
    batch.flush()?;
    batch
        .timing
        .as_mut()
        .unwrap()
        .enter("smoke_finished", false);
    let mut out = vec![0; values.len() * 4];
    arena.transfer("out", &mut out, false)?;
    assert_eq!(synapse_core::decode_f32_frame(&out)?, values);
    Ok(())
}
