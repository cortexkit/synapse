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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ane_subtype: Option<String>,
}

pub trait MachineProfileCollector {
    fn collect_base_profile(&self) -> MachineProfileBase;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemMachineProfileCollector;

impl MachineProfileCollector for SystemMachineProfileCollector {
    fn collect_base_profile(&self) -> MachineProfileBase {
        let chip_model = chip_model();
        MachineProfileBase {
            os_build: os_build(),
            arch: std::env::consts::ARCH.to_string(),
            chip_model: chip_model.clone(),
            ram_class: ram_class(),
            ane_subtype: ane_subtype(&chip_model),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineProfile {
    pub os_build: String,
    pub arch: String,
    pub chip_model: String,
    pub ram_class: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ane_subtype: Option<String>,
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
            ane_subtype: base.ane_subtype,
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

    /// Hash used by new certification records. The explicit revision prevents
    /// future profile-field additions from silently reusing old evidence.
    #[must_use]
    pub fn revisioned_hash(&self) -> String {
        crate::revisioned_machine_profile_hash(&self.stable_bytes())
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

fn ane_subtype(chip_model: &str) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        // Public IORegistry inspection on current Apple silicon exposes ANE
        // firmware functions but no stable subtype property. Keep private
        // _ANEDeviceInfo out of the daemon and use the static chip-identity
        // mapping until macOS exposes a supported read-only subtype value.
        mapped_ane_subtype(chip_model)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = chip_model;
        None
    }
}

#[cfg(any(target_os = "macos", test))]
fn mapped_ane_subtype(chip_model: &str) -> Option<String> {
    let chip_model = chip_model.trim().to_ascii_lowercase();
    if chip_model == "apple m5 max" {
        Some("h17(map)".to_string())
    } else if chip_model == "apple m5"
        || chip_model == "apple m4"
        || chip_model.starts_with("apple m4 ")
    {
        Some("h16(map)".to_string())
    } else {
        None
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

    struct FakeCollector {
        ane_subtype: Option<&'static str>,
    }

    impl MachineProfileCollector for FakeCollector {
        fn collect_base_profile(&self) -> MachineProfileBase {
            MachineProfileBase {
                os_build: "23G93".to_string(),
                arch: "aarch64".to_string(),
                chip_model: "Apple M3".to_string(),
                ram_class: "le_32_gib".to_string(),
                ane_subtype: self.ane_subtype.map(str::to_string),
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

        let collector = FakeCollector { ane_subtype: None };
        let left = MachineProfile::collect(&collector, [ort.clone(), llama.clone()]);
        let right = MachineProfile::collect(&collector, [llama, ort]);
        assert_eq!(left, right);
        assert_eq!(left.hash(), right.hash());
        assert_eq!(left.engine_identities[0].engine, "llama.cpp");
    }

    #[test]
    fn ane_subtype_mapping_marks_chip_identity_provenance() {
        assert_eq!(
            mapped_ane_subtype("Apple M5 Max"),
            Some("h17(map)".to_string())
        );
        assert_eq!(mapped_ane_subtype("Apple M5"), Some("h16(map)".to_string()));
        assert_eq!(
            mapped_ane_subtype("Apple M4 Max"),
            Some("h16(map)".to_string())
        );
        assert_eq!(mapped_ane_subtype("Apple M3 Max"), None);
    }

    #[test]
    fn ane_subtype_changes_profile_hash_and_none_keeps_legacy_shape() {
        let without_ane = MachineProfile::collect(
            &FakeCollector { ane_subtype: None },
            std::iter::empty::<EngineIdentity>(),
        );
        let with_ane = MachineProfile::collect(
            &FakeCollector {
                ane_subtype: Some("h17(map)"),
            },
            std::iter::empty::<EngineIdentity>(),
        );

        assert_ne!(without_ane.hash(), with_ane.hash());
        assert_eq!(
            without_ane.hash(),
            "883d3caf3aa4da4277fe8744fefa4829ee9d1c00bde0722c41d6b6ce959427c0"
        );
        assert_eq!(without_ane.ane_subtype, None);
        assert_eq!(with_ane.ane_subtype.as_deref(), Some("h17(map)"));
        assert!(!String::from_utf8(without_ane.stable_bytes())
            .unwrap()
            .contains("ane_subtype"));
    }

    #[test]
    fn ram_class_buckets_to_stable_labels() {
        assert_eq!(ram_class_from_bytes(7 * 1024 * 1024 * 1024), "le_8_gib");
        assert_eq!(ram_class_from_bytes(300 * 1024 * 1024 * 1024), "gt_256_gib");
    }
}
