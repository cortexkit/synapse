//! Immutable model contracts shared by the worker's probe and loader.
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::LazyLock,
};

pub const MANIFEST_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/models.json"));
pub static MANIFEST: LazyLock<Value> =
    LazyLock::new(|| serde_json::from_slice(MANIFEST_BYTES).expect("embedded manifest"));
pub fn manifest_digest() -> String {
    env!("SYNAPSE_CUDA_MANIFEST_DIGEST").into()
}
pub fn max_context_tokens() -> usize {
    MANIFEST["admission"]["max_context_tokens"]
        .as_u64()
        .expect("manifest context limit") as usize
}

#[derive(Clone, Debug)]
pub struct Profile {
    pub id: String,
    pub model: Value,
    pub package_digest: String,
    pub floor: HardwareFloor,
}
#[derive(Clone, Copy, Debug)]
pub struct HardwareFloor {
    pub driver_api: u32,
    pub compute_major: u32,
    pub compute_minor: u32,
}
impl HardwareFloor {
    fn from_profile(profile: &Value) -> Self {
        let integer = |name: &str| {
            u32::try_from(profile[name].as_u64().expect("required CUDA floor integer"))
                .expect("CUDA floor fits u32")
        };
        Self {
            driver_api: integer("cuda_min_driver_api"),
            compute_major: integer("cuda_min_compute_major"),
            compute_minor: integer("cuda_min_compute_minor"),
        }
    }
    pub fn accepts(self, observed: crate::HardwareFloorProbe) -> bool {
        observed.driver_api >= self.driver_api
            && (observed.compute_major, observed.compute_minor)
                >= (self.compute_major, self.compute_minor)
    }
}
pub fn default_floor() -> HardwareFloor {
    HardwareFloor::from_profile(&MANIFEST["profiles"]["gte-modernbert-base.owned-cuda"])
}
impl Profile {
    pub fn select(id: &str, operation: Option<&str>) -> Result<Self> {
        let profile = &MANIFEST["profiles"][id];
        if profile["lane"] != "owned-cuda" {
            bail!("model_unsupported");
        }
        let slug = profile["model"].as_str().context("model_unsupported")?;
        let model = MANIFEST["models"][slug].clone();
        if operation.is_some_and(|op| model["operation"] != op) {
            bail!("operation_mismatch");
        }
        Ok(Self {
            id: id.into(),
            model,
            floor: HardwareFloor::from_profile(profile),
            package_digest: profile["converted_package_digest"]
                .as_str()
                .context("package_digest_mismatch")?
                .into(),
        })
    }
    pub fn operation(&self) -> &str {
        self.model["operation"]
            .as_str()
            .expect("manifest operation")
    }
    pub fn pad_id(&self) -> u32 {
        self.model["grammar"]["pad"]["id"]
            .as_u64()
            .expect("manifest pad") as u32
    }
    pub fn params(&self) -> Value {
        self.model["architecture"]["params"].clone()
    }
    pub fn package_path(path: &Path) -> PathBuf {
        if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.into()
        }
    }
    /// Validate head metadata before mapping or decoding any weight tensor.
    pub fn validate_header(&self, path: &Path, digest: &str) -> Result<PathBuf> {
        if digest != self.package_digest {
            bail!("package_digest_mismatch");
        }
        let path = Self::package_path(path);
        let mut file = std::fs::File::open(&path).context("artifact_invalid")?;
        let mut size = [0; 8];
        file.read_exact(&mut size).context("artifact_invalid")?;
        let size = u64::from_le_bytes(size);
        if size > 16 * 1024 * 1024 {
            bail!("artifact_invalid");
        }
        let mut bytes = vec![0; size as usize];
        file.read_exact(&mut bytes).context("artifact_invalid")?;
        let header: Value = serde_json::from_slice(&bytes).context("artifact_invalid")?;
        if header["__metadata__"]["profile"] != self.id {
            bail!("package_digest_mismatch");
        }
        if let Some(tensors) = self.model["head"]["tensors"].as_object() {
            for tensor in tensors.values() {
                let key = tensor["key"].as_str().context("head_tensor_missing")?;
                if header[key]["shape"] != tensor["shape"] {
                    bail!("head_tensor_missing");
                }
            }
        }
        if let Some(keys) = self.model["head"]["forbidden_tensor_keys"].as_array() {
            if keys
                .iter()
                .any(|key| header.get(key.as_str().unwrap()).is_some())
            {
                bail!("head_tensor_missing");
            }
        }
        Ok(path)
    }
    pub fn verify_package(&self, path: &Path) -> Result<()> {
        let mut file = std::fs::File::open(path)?;
        let mut hash = Sha256::new();
        std::io::copy(&mut file, &mut hash)?;
        if format!("sha256:{:x}", hash.finalize()) != self.package_digest {
            bail!("package_digest_mismatch");
        }
        Ok(())
    }
}

pub fn floor_envelope(observed: Result<crate::HardwareFloorProbe>, model: Option<&str>) -> Value {
    let profile = Profile::select(
        &format!("{}.owned-cuda", model.unwrap_or("gte-modernbert-base")),
        None,
    );
    match profile {
        Ok(profile) => floor_envelope_for_profile(observed, &profile),
        Err(_) => {
            json!({"status":"refused","code":"model_unsupported","required":null,"observed":null})
        }
    }
}
pub fn floor_envelope_for_profile(
    observed: Result<crate::HardwareFloorProbe>,
    profile: &Profile,
) -> Value {
    let floor = profile.floor;
    let required = json!({"driver_api":floor.driver_api,"compute_capability":{"major":floor.compute_major,"minor":floor.compute_minor}});
    match observed {
        Ok(p) => {
            let code = if p.driver_api < floor.driver_api {
                "cuda_driver_too_old"
            } else if (p.compute_major, p.compute_minor)
                < (floor.compute_major, floor.compute_minor)
            {
                "cuda_compute_capability_too_low"
            } else {
                "ok"
            };
            json!({"status":if code=="ok" {"ok"} else {"refused"},"code":code,"required":required,"observed":{"driver_api":p.driver_api,"compute_capability":{"major":p.compute_major,"minor":p.compute_minor}}})
        }
        Err(error) => {
            let message = error.to_string();
            let code = if message.starts_with("cuda_runtime_missing:") {
                message.as_str()
            } else {
                "cuda_no_driver"
            };
            json!({"status":"refused","code":code,"required":required,"observed":null})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floor_boundary_is_typed() {
        for (driver_api, code) in [(13019, "cuda_driver_too_old"), (13020, "ok")] {
            let envelope = floor_envelope(
                Ok(crate::HardwareFloorProbe {
                    driver_api,
                    compute_major: 7,
                    compute_minor: 5,
                }),
                None,
            );
            assert_eq!(envelope["code"], code);
            assert_eq!(envelope["required"]["driver_api"], 13020);
        }
    }
    #[test]
    fn profile_refusals_precede_artifact_access() {
        assert_eq!(
            Profile::select("unknown", None).unwrap_err().to_string(),
            "model_unsupported"
        );
        assert_eq!(
            Profile::select("gte-modernbert-base.owned-cuda", Some("rerank"))
                .unwrap_err()
                .to_string(),
            "operation_mismatch"
        );
        let p = Profile::select("gte-modernbert-base.owned-cuda", Some("embed")).unwrap();
        assert_eq!(
            p.validate_header(Path::new("does-not-exist"), "wrong")
                .unwrap_err()
                .to_string(),
            "package_digest_mismatch"
        );
    }
}

#[cfg(test)]
mod header_tests {
    use super::*;
    #[test]
    fn missing_head_is_refused_from_header_only() {
        let profile =
            Profile::select("gte-reranker-modernbert-base.owned-cuda", Some("rerank")).unwrap();
        let path = std::env::temp_dir().join(format!(
            "cuda-missing-head-{}.safetensors",
            std::process::id()
        ));
        let header = serde_json::to_vec(
            &json!({"__metadata__":{"profile":profile.id,"conversion_rule":"v1"}}),
        )
        .unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(
            profile
                .validate_header(&path, &profile.package_digest)
                .unwrap_err()
                .to_string(),
            "head_tensor_missing"
        );
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod selected_floor_tests {
    use super::*;
    #[test]
    fn selected_profile_integers_drive_probe_requirement_and_refusal() {
        let mut profile = Profile::select("gte-modernbert-base.owned-cuda", None).unwrap();
        let mut table = MANIFEST["profiles"][&profile.id].clone();
        table["cuda_min_driver_api"] = 14000.into();
        table["cuda_min_compute_major"] = 9.into();
        table["cuda_min_compute_minor"] = 1.into();
        profile.floor = HardwareFloor::from_profile(&table);
        let probe = crate::HardwareFloorProbe {
            driver_api: 14000,
            compute_major: 9,
            compute_minor: 0,
        };
        let envelope = floor_envelope_for_profile(Ok(probe), &profile);
        assert_eq!(envelope["required"]["driver_api"], 14000);
        assert_eq!(
            envelope["required"]["compute_capability"],
            json!({"major":9,"minor":1})
        );
        assert_eq!(envelope["code"], "cuda_compute_capability_too_low");
        assert_eq!(
            floor_envelope_for_profile(
                Ok(crate::HardwareFloorProbe {
                    compute_minor: 1,
                    ..probe
                }),
                &profile
            )["code"],
            "ok"
        );
    }
}
