use serde::{Deserialize, Serialize};
use synapse_parity::manifest::{Lane, Manifest};
use synapse_parity::vulkan::{vulkan_floors, FLOOR_SEQUENCES};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Required {
    pub min_storage_buffer_range: u64,
    pub min_device_local_bytes: u64,
}

pub fn requirements(manifest: &Manifest, slug: Option<&str>) -> Result<Required, String> {
    let mut required = Required {
        min_storage_buffer_range: 0,
        min_device_local_bytes: 0,
    };
    let mut found = false;
    for profile in manifest
        .profiles
        .values()
        .filter(|p| p.lane == Lane::OwnedVulkan && slug.is_none_or(|s| p.model == s))
    {
        let model = &manifest.models[&profile.model];
        let floors = vulkan_floors(
            model,
            profile.storage_dtype,
            &profile.fp32_tensors,
            u64::from(manifest.admission.max_context_tokens),
            FLOOR_SEQUENCES,
            u64::from(
                profile
                    .vulkan_sub_batch_max_tokens
                    .ok_or("model_unsupported")?,
            ),
        )
        .map_err(|e| e.to_string())?;
        if Some(floors.min_storage_buffer_range) != profile.vulkan_min_storage_buffer_range
            || Some(floors.min_device_local_bytes) != profile.vulkan_min_device_local_bytes
        {
            return Err("manifest_mismatch".into());
        }
        required.min_storage_buffer_range = required
            .min_storage_buffer_range
            .max(floors.min_storage_buffer_range);
        required.min_device_local_bytes = required
            .min_device_local_bytes
            .max(floors.min_device_local_bytes);
        found = true;
    }
    if found {
        Ok(required)
    } else {
        Err("model_unsupported".into())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adapter {
    pub index: usize,
    pub discrete: bool,
    pub cpu: bool,
    pub vendor: u32,
    pub api_major: u32,
    pub api_minor: u32,
    pub shader_float16: bool,
    pub storage_buffer16_bit_access: bool,
    pub subgroup_arithmetic: bool,
    pub subgroup_compute_stage: bool,
    pub max_storage_buffer_range: u64,
    pub device_local_heaps: Vec<u64>,
    pub cooperative_matrix: bool,
}

impl Adapter {
    pub fn check(&self, required: Required) -> Result<(), String> {
        if self.cpu {
            return Err("vulkan_software_device".into());
        }
        if ![0x1002, 0x10de].contains(&self.vendor) {
            return Err("vulkan_unsupported_vendor".into());
        }
        if (self.api_major, self.api_minor) < (1, 2) {
            return Err("vulkan_api_too_old".into());
        }
        for (name, present) in [
            ("shaderFloat16", self.shader_float16),
            ("storageBuffer16BitAccess", self.storage_buffer16_bit_access),
            ("subgroupArithmetic", self.subgroup_arithmetic),
            ("subgroupComputeStage", self.subgroup_compute_stage),
        ] {
            if !present {
                return Err(format!("vulkan_missing_feature:{name}"));
            }
        }
        if self.max_storage_buffer_range < required.min_storage_buffer_range
            || self.device_local_heaps.iter().copied().max().unwrap_or(0)
                < required.min_device_local_bytes
        {
            return Err("vulkan_insufficient_memory".into());
        }
        Ok(())
    }

    pub fn use_cooperative(&self) -> bool {
        (self.api_major, self.api_minor) >= (1, 3) && self.cooperative_matrix
    }
}

pub fn select(mut devices: Vec<Adapter>, required: Required) -> Result<Adapter, String> {
    devices.sort_by_key(|d| (!d.discrete, d.index));
    let mut first = None;
    for device in devices {
        match device.check(required) {
            Ok(()) => return Ok(device),
            Err(code) => {
                first.get_or_insert(code);
            }
        }
    }
    Err(first.unwrap_or_else(|| "vulkan_no_device".into()))
}

#[derive(Debug, Serialize)]
pub struct Probe {
    pub status: &'static str,
    pub code: Option<String>,
    pub required: Required,
    pub observed: Option<Adapter>,
}

impl Probe {
    pub fn exit_code(&self) -> i32 {
        if self.status == "ok" {
            0
        } else {
            2
        }
    }
}

pub fn probe(required: Required, lookup: impl FnOnce() -> Result<Vec<Adapter>, String>) -> Probe {
    match lookup() {
        Ok(mut devices) => {
            devices.sort_by_key(|d| (!d.discrete, d.index));
            let first = devices.first().cloned();
            match select(devices, required) {
                Ok(device) => Probe {
                    status: "ok",
                    code: None,
                    required,
                    observed: Some(device),
                },
                Err(code) => Probe {
                    status: "refused",
                    code: Some(code),
                    required,
                    observed: first,
                },
            }
        }
        Err(code) => Probe {
            status: "refused",
            code: Some(code),
            required,
            observed: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn device() -> Adapter {
        Adapter {
            index: 0,
            discrete: true,
            cpu: false,
            vendor: 0x1002,
            api_major: 1,
            api_minor: 2,
            shader_float16: true,
            storage_buffer16_bit_access: true,
            subgroup_arithmetic: true,
            subgroup_compute_stage: true,
            max_storage_buffer_range: u64::MAX,
            device_local_heaps: vec![1, u64::MAX],
            cooperative_matrix: true,
        }
    }
    #[test]
    fn ordered_vendor_refusal_and_selection() {
        let required = requirements(&crate::manifest(), None).unwrap();
        let mut intel = device();
        intel.vendor = 0x8086;
        let mut cpu = device();
        cpu.cpu = true;
        cpu.discrete = false;
        cpu.index = 1;
        let envelope = probe(required, || Ok(vec![intel.clone()]));
        assert_eq!(envelope.code.as_deref(), Some("vulkan_unsupported_vendor"));
        assert_eq!(envelope.exit_code(), 2);
        assert_eq!(
            select(vec![intel.clone()], required).unwrap_err(),
            "vulkan_unsupported_vendor"
        );
        let mixed = probe(required, || Ok(vec![cpu, intel]));
        assert_eq!(mixed.code.as_deref(), Some("vulkan_unsupported_vendor"));
        assert_eq!(mixed.exit_code(), 2);
        let empty = probe(required, || Ok(Vec::new()));
        assert_eq!(empty.code.as_deref(), Some("vulkan_no_device"));
        assert_eq!(empty.exit_code(), 2);
        let mut a = device();
        a.index = 4;
        assert_eq!(select(vec![a, device()], required).unwrap().index, 0);
    }
    #[test]
    fn each_missing_requirement_has_exact_code() {
        let required = requirements(&crate::manifest(), None).unwrap();
        let cases: Vec<(&str, Box<dyn Fn(&mut Adapter)>)> = vec![
            (
                "vulkan_software_device",
                Box::new(|a| {
                    a.cpu = true;
                    a.vendor = 0x8086;
                }),
            ),
            ("vulkan_unsupported_vendor", Box::new(|a| a.vendor = 0x8086)),
            ("vulkan_api_too_old", Box::new(|a| a.api_minor = 1)),
            (
                "vulkan_missing_feature:shaderFloat16",
                Box::new(|a| a.shader_float16 = false),
            ),
            (
                "vulkan_missing_feature:storageBuffer16BitAccess",
                Box::new(|a| a.storage_buffer16_bit_access = false),
            ),
            (
                "vulkan_missing_feature:subgroupArithmetic",
                Box::new(|a| a.subgroup_arithmetic = false),
            ),
            (
                "vulkan_missing_feature:subgroupComputeStage",
                Box::new(|a| a.subgroup_compute_stage = false),
            ),
        ];
        for (code, mutate) in cases {
            let mut a = device();
            mutate(&mut a);
            assert_eq!(a.check(required).unwrap_err(), code);
        }
    }
    #[test]
    fn pinned_byte_floors_use_largest_heap_not_sum() {
        for slug in synapse_parity::manifest::MODEL_SLUGS {
            let required = requirements(&crate::manifest(), Some(slug)).unwrap();
            let mut a = device();
            a.max_storage_buffer_range = required.min_storage_buffer_range;
            a.device_local_heaps = vec![
                required.min_device_local_bytes / 2,
                required.min_device_local_bytes,
            ];
            assert!(a.check(required).is_ok());
            a.max_storage_buffer_range -= 1;
            assert_eq!(a.check(required).unwrap_err(), "vulkan_insufficient_memory");
            a.max_storage_buffer_range += 1;
            a.device_local_heaps[1] -= 1;
            assert_eq!(a.check(required).unwrap_err(), "vulkan_insufficient_memory");
        }
    }
    #[test]
    fn cooperative_requires_both_api_and_extension() {
        let mut a = device();
        assert!(!a.use_cooperative());
        a.api_minor = 3;
        assert!(a.use_cooperative());
        a.cooperative_matrix = false;
        assert!(!a.use_cooperative());
    }
}
