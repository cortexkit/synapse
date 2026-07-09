use std::process::Command;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::EngineIdentity;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineProfileBase {
    pub os_build: String,
    pub arch: String,
    pub chip_model: String,
    pub ram_class: String,
}

pub trait MachineProfileCollector {
    fn collect_base_profile(&self) -> MachineProfileBase;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemMachineProfileCollector;

impl MachineProfileCollector for SystemMachineProfileCollector {
    fn collect_base_profile(&self) -> MachineProfileBase {
        MachineProfileBase {
            os_build: os_build(),
            arch: std::env::consts::ARCH.to_string(),
            chip_model: chip_model(),
            ram_class: ram_class(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineProfile {
    pub os_build: String,
    pub arch: String,
    pub chip_model: String,
    pub ram_class: String,
    #[serde(default)]
    pub engine_identities: Vec<EngineIdentity>,
}

impl MachineProfile {
    pub fn collect<C, I>(collector: &C, engine_identities: I) -> Self
    where
        C: MachineProfileCollector,
        I: IntoIterator<Item = EngineIdentity>,
    {
        let base = collector.collect_base_profile();
        let mut engine_identities = engine_identities.into_iter().collect::<Vec<_>>();
        engine_identities.sort_by(|left, right| {
            left.engine
                .cmp(&right.engine)
                .then_with(|| left.version.cmp(&right.version))
                .then_with(|| left.build_flags.cmp(&right.build_flags))
        });
        Self {
            os_build: base.os_build,
            arch: base.arch,
            chip_model: base.chip_model,
            ram_class: base.ram_class,
            engine_identities,
        }
    }

    #[must_use]
    pub fn stable_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("machine profile should always serialize")
    }

    #[must_use]
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.stable_bytes());
        hex::encode(hasher.finalize())
    }
}

fn os_build() -> String {
    #[cfg(target_os = "macos")]
    {
        command_stdout("sw_vers", &["-buildVersion"])
            .unwrap_or_else(|| format!("{}-unknown", std::env::consts::OS))
    }
    #[cfg(not(target_os = "macos"))]
    {
        command_stdout("uname", &["-sr"])
            .unwrap_or_else(|| format!("{}-unknown", std::env::consts::OS))
    }
}

fn chip_model() -> String {
    #[cfg(target_os = "macos")]
    {
        sysctl_value("machdep.cpu.brand_string")
            .or_else(|| sysctl_value("hw.model"))
            .unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        command_stdout("uname", &["-m"]).unwrap_or_else(|| std::env::consts::ARCH.to_string())
    }
}

fn ram_class() -> String {
    #[cfg(target_os = "macos")]
    {
        sysctl_value("hw.memsize")
            .and_then(|value| value.parse::<u64>().ok())
            .map(ram_class_from_bytes)
            .unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        "unknown".to_string()
    }
}

#[cfg(any(target_os = "macos", test))]
fn ram_class_from_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    let gib = bytes.div_ceil(GIB).max(1);
    for bucket in [4_u64, 8, 16, 32, 64, 128, 256] {
        if gib <= bucket {
            return format!("le_{bucket}_gib");
        }
    }
    "gt_256_gib".to_string()
}

#[cfg(target_os = "macos")]
fn sysctl_value(name: &str) -> Option<String> {
    command_stdout("sysctl", &["-n", name])
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    struct FakeCollector;

    impl MachineProfileCollector for FakeCollector {
        fn collect_base_profile(&self) -> MachineProfileBase {
            MachineProfileBase {
                os_build: "23G93".to_string(),
                arch: "aarch64".to_string(),
                chip_model: "Apple M3".to_string(),
                ram_class: "le_32_gib".to_string(),
            }
        }
    }

    #[test]
    fn machine_profile_hash_is_stable_and_sorts_engines() {
        let mut flags = BTreeMap::new();
        flags.insert("execution_provider".to_string(), "cpu".to_string());
        let ort = EngineIdentity {
            engine: "ort".to_string(),
            version: "2.0".to_string(),
            build_flags: flags,
        };
        let llama = EngineIdentity {
            engine: "llama.cpp".to_string(),
            version: "1.0".to_string(),
            build_flags: BTreeMap::new(),
        };

        let left = MachineProfile::collect(&FakeCollector, [ort.clone(), llama.clone()]);
        let right = MachineProfile::collect(&FakeCollector, [llama, ort]);
        assert_eq!(left, right);
        assert_eq!(left.hash(), right.hash());
        assert_eq!(left.engine_identities[0].engine, "llama.cpp");
    }

    #[test]
    fn ram_class_buckets_to_stable_labels() {
        assert_eq!(ram_class_from_bytes(7 * 1024 * 1024 * 1024), "le_8_gib");
        assert_eq!(ram_class_from_bytes(300 * 1024 * 1024 * 1024), "gt_256_gib");
    }
}
