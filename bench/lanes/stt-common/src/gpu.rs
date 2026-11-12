use std::collections::BTreeMap;
use std::process::Command;

use crate::{memory::Memory, nvml::Nvml};

type Devices = BTreeMap<String, u64>;

const PID_NVML: &str =
    "NVML v2 compute+graphics process lists; deduplicated per device, summed across devices";
const PID_SMI: &str = "nvidia-smi --query-compute-apps=pid,used_gpu_memory; sum matching PID across devices; MiB to bytes";
const DEVICE_NVML: &str = "device-delta (per-process unavailable, including WDDM N/A); NVML nvmlDeviceGetMemoryInfo, all NVIDIA GPUs; includes other GPU users";
const DEVICE_SMI: &str = "device-delta (per-process unavailable, including WDDM N/A); nvidia-smi --query-gpu=uuid,memory.used, all NVIDIA GPUs; includes other GPU users; MiB to bytes";

enum DeviceSource {
    Nvml,
    Smi,
}

/// Capture device memory before launching the worker. A missing baseline can
/// never be replaced with a post-spawn reading that already includes weights.
pub struct GpuProbe {
    nvml: Option<Nvml>,
    device_source: DeviceSource,
    readings: Readings,
}

impl GpuProbe {
    pub fn before_spawn() -> Self {
        let nvml = Nvml::open();
        let baseline = nvml.as_ref().and_then(Nvml::device_bytes);
        let (baseline, device_source) = if baseline.is_some() {
            (baseline, DeviceSource::Nvml)
        } else {
            (query_devices(), DeviceSource::Smi)
        };
        Self {
            nvml,
            device_source,
            readings: Readings {
                baseline,
                ..Default::default()
            },
        }
    }

    pub(crate) fn sample(&mut self, pid: u32, loaded: bool, memory: &mut Memory) {
        let process = if let Some(nvml) = &self.nvml {
            nvml.bytes(pid)
        } else {
            query_process(pid)
        };
        let devices = if process.is_none() {
            match self.device_source {
                DeviceSource::Nvml => self.nvml.as_ref().and_then(Nvml::device_bytes),
                DeviceSource::Smi => query_devices(),
            }
        } else {
            None
        };
        self.readings.record(process, devices.as_ref(), loaded);
        self.readings.apply(
            memory,
            if self.nvml.is_some() {
                PID_NVML
            } else {
                PID_SMI
            },
            match self.device_source {
                DeviceSource::Nvml => DEVICE_NVML,
                DeviceSource::Smi => DEVICE_SMI,
            },
        );
    }
}

#[derive(Default)]
struct Readings {
    baseline: Option<Devices>,
    process_peak: Option<u64>,
    device_peak: Option<u64>,
    process_idle: Option<u64>,
    device_idle: Option<u64>,
    process_samples: u64,
    device_samples: u64,
}

impl Readings {
    fn record(&mut self, process: Option<u64>, devices: Option<&Devices>, loaded: bool) {
        let delta = self
            .baseline
            .as_ref()
            .zip(devices)
            .and_then(|(baseline, current)| {
                // GPU UUIDs must match: a different device set is not a valid delta.
                if !baseline.keys().eq(current.keys()) {
                    return None;
                }
                Some(total(current)?.saturating_sub(total(baseline)?))
            });
        if let Some(bytes) = process {
            self.process_peak = Some(self.process_peak.unwrap_or(0).max(bytes));
            self.process_samples += 1;
        }
        if let Some(bytes) = delta {
            self.device_peak = Some(self.device_peak.unwrap_or(0).max(bytes));
            self.device_samples += 1;
        }
        if loaded {
            self.process_idle = process;
            self.device_idle = delta;
        }
    }

    fn apply(&self, memory: &mut Memory, process_method: &str, device_method: &str) {
        memory.gpu_device_baseline_bytes = self.baseline.as_ref().and_then(total);
        // Keep scopes separate: a valid per-PID sample always takes priority
        // over startup device samples, rather than mixing their peaks.
        memory.gpu_peak_bytes = self.process_peak.or(self.device_peak);
        memory.gpu_peak_method = if self.process_peak.is_some() {
            Some(process_method)
        } else if self.device_peak.is_some() {
            Some(device_method)
        } else {
            None
        }
        .map(str::to_owned);
        memory.gpu_idle_baseline_bytes = if self.process_peak.is_some() {
            Some(0)
        } else if self.device_peak.is_some() {
            memory.gpu_device_baseline_bytes
        } else {
            None
        };
        memory.gpu_samples = if self.process_peak.is_some() {
            self.process_samples
        } else {
            self.device_samples
        };
        memory.gpu_after_load_idle_bytes = self.process_idle.or(self.device_idle);
        memory.gpu_after_load_idle_method = if self.process_idle.is_some() {
            Some(process_method)
        } else if self.device_idle.is_some() {
            Some(device_method)
        } else {
            None
        }
        .map(str::to_owned);
    }
}

fn total(devices: &Devices) -> Option<u64> {
    if devices.is_empty() {
        return None;
    }
    devices
        .values()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))
}

fn smi(args: &[&str]) -> Option<String> {
    let output = Command::new("nvidia-smi").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

fn query_process(pid: u32) -> Option<u64> {
    parse_process(
        &smi(&[
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])?,
        pid,
    )
}

fn parse_process(text: &str, pid: u32) -> Option<u64> {
    let mut total: Option<u64> = None;
    for line in text.lines() {
        let (process, mib) = line.split_once(',')?;
        if process.trim().parse::<u32>().ok()? == pid {
            let bytes = mib.trim().parse::<u64>().ok()?.checked_mul(1024 * 1024)?;
            total = Some(total.unwrap_or(0).checked_add(bytes)?);
        }
    }
    total
}

fn query_devices() -> Option<Devices> {
    parse_devices(&smi(&[
        "--query-gpu=uuid,memory.used",
        "--format=csv,noheader,nounits",
    ])?)
}

fn parse_devices(text: &str) -> Option<Devices> {
    let mut devices = Devices::new();
    for line in text.lines() {
        let (uuid, mib) = line.split_once(',')?;
        let bytes = mib.trim().parse::<u64>().ok()?.checked_mul(1024 * 1024)?;
        if uuid.trim().is_empty() || devices.insert(uuid.trim().to_owned(), bytes).is_some() {
            return None;
        }
    }
    total(&devices)?;
    Some(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn devices(values: &[(&str, u64)]) -> Devices {
        values
            .iter()
            .map(|(id, bytes)| ((*id).into(), *bytes))
            .collect()
    }

    #[test]
    fn stt_memory_wddm_fallback_subtracts_pre_spawn_device_baseline() {
        let mut readings = Readings {
            baseline: Some(devices(&[("gpu-a", 200), ("gpu-b", 300)])),
            ..Default::default()
        };
        readings.record(
            None,
            Some(&devices(&[("gpu-a", 250), ("gpu-b", 450)])),
            true,
        );
        readings.record(
            None,
            Some(&devices(&[("gpu-a", 300), ("gpu-b", 700)])),
            false,
        );
        let mut memory = Memory::default();
        readings.apply(&mut memory, PID_NVML, DEVICE_NVML);
        assert_eq!(memory.gpu_peak_bytes, Some(500));
        assert_eq!(memory.gpu_after_load_idle_bytes, Some(200));
        assert_eq!(memory.gpu_idle_baseline_bytes, Some(500));
        assert_eq!(memory.gpu_device_baseline_bytes, Some(500));
        assert!(memory
            .gpu_peak_method
            .as_deref()
            .unwrap()
            .contains("device-delta"));
        assert!(memory
            .gpu_after_load_idle_method
            .as_deref()
            .unwrap()
            .contains("other GPU users"));
        assert_eq!(memory.gpu_samples, 2);
        readings.record(Some(0), None, false);
        readings.apply(&mut memory, PID_NVML, DEVICE_NVML);
        assert_eq!(memory.gpu_peak_bytes, Some(0));
        assert_eq!(memory.gpu_peak_method.as_deref(), Some(PID_NVML));
    }

    #[test]
    fn stt_memory_per_pid_readings_take_priority_over_device_deltas() {
        let mut readings = Readings {
            baseline: Some(devices(&[("gpu", 100)])),
            ..Default::default()
        };
        readings.record(None, Some(&devices(&[("gpu", 1000)])), false);
        readings.record(Some(40), None, true);
        readings.record(Some(70), None, false);
        let mut memory = Memory::default();
        readings.apply(&mut memory, PID_NVML, DEVICE_NVML);
        assert_eq!(memory.gpu_peak_bytes, Some(70));
        assert_eq!(memory.gpu_after_load_idle_bytes, Some(40));
        assert_eq!(memory.gpu_idle_baseline_bytes, Some(0));
        assert_eq!(memory.gpu_peak_method.as_deref(), Some(PID_NVML));
        assert_eq!(memory.gpu_after_load_idle_method.as_deref(), Some(PID_NVML));
        assert_eq!(memory.gpu_samples, 2);
    }

    #[test]
    fn stt_memory_device_fallback_requires_same_devices_and_clamps_releases() {
        let mut readings = Readings::default();
        readings.record(None, Some(&devices(&[("gpu", 10)])), true);
        let mut memory = Memory::default();
        readings.apply(&mut memory, PID_NVML, DEVICE_NVML);
        assert_eq!(memory.gpu_peak_bytes, None);
        readings.baseline = Some(devices(&[("gpu", 100)]));
        readings.record(None, Some(&devices(&[("different-gpu", 500)])), true);
        readings.apply(&mut memory, PID_NVML, DEVICE_NVML);
        assert_eq!(memory.gpu_peak_bytes, None);
        readings.record(None, Some(&devices(&[("gpu", 90)])), true);
        readings.apply(&mut memory, PID_NVML, DEVICE_NVML);
        assert_eq!(memory.gpu_peak_bytes, Some(0));
        assert_eq!(memory.gpu_after_load_idle_bytes, Some(0));
        readings.record(Some(20), None, false);
        readings.apply(&mut memory, PID_NVML, DEVICE_NVML);
        assert_eq!(memory.gpu_peak_method.as_deref(), Some(PID_NVML));
        assert_eq!(
            memory.gpu_after_load_idle_method.as_deref(),
            Some(DEVICE_NVML)
        );
    }

    #[test]
    fn stt_memory_smi_parsers_keep_units_identity_and_unavailable_values() {
        assert_eq!(
            parse_process("42, 10\n99, 999\n42, 2\n", 42),
            Some(12 * 1024 * 1024)
        );
        assert_eq!(parse_process("42, 0\n", 42), Some(0));
        assert_eq!(parse_process("42, [N/A]\n", 42), None);
        assert_eq!(parse_process("99, 200\n", 42), None);
        assert_eq!(parse_process("", 42), None);
        assert_eq!(
            parse_devices("gpu-b, 3\ngpu-a, 2\n"),
            Some(devices(&[
                ("gpu-a", 2 * 1024 * 1024),
                ("gpu-b", 3 * 1024 * 1024)
            ]))
        );
        assert_eq!(parse_devices("gpu, N/A\n"), None);
        assert_eq!(parse_devices("gpu, 1\ngpu, 2\n"), None);
        assert_eq!(parse_devices(""), None);
    }
}
