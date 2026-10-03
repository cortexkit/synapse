use std::collections::BTreeMap;
use synapse_parity::vulkan::{aligned_bytes, arena_bytes, Buffer};

/// One arena per model; all views share it and every sub-batch reuses the views.
/// The driver allocation requirement is reported separately by the Vulkan runtime.
#[derive(Debug)]
pub struct Layout {
    pub slices: BTreeMap<String, (u64, u64)>,
    pub requested_bytes: u64,
}
impl Layout {
    pub fn new(plan: &[Buffer], cap: u64) -> Result<Self, String> {
        let requested_bytes = arena_bytes(plan);
        if requested_bytes > cap {
            return Err("vulkan_insufficient_memory".into());
        }
        let mut offset = 0;
        let slices = plan
            .iter()
            .map(|buffer| {
                let slice = (offset, buffer.bytes);
                offset += aligned_bytes(buffer.bytes);
                (buffer.name.clone(), slice)
            })
            .collect();
        Ok(Self {
            slices,
            requested_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_parity::{
        manifest::Lane,
        vulkan::{buffer_plan, BUFFER_ALIGNMENT, FLOOR_SEQUENCES},
    };
    #[test]
    fn allocator_reuses_one_arena_for_256_max_context_sequences() {
        let manifest = crate::manifest();
        for profile in manifest
            .profiles
            .values()
            .filter(|p| p.lane == Lane::OwnedVulkan)
        {
            let plan = buffer_plan(
                &manifest.models[&profile.model],
                profile.storage_dtype,
                &profile.fp32_tensors,
                8192,
                FLOOR_SEQUENCES,
                u64::from(profile.vulkan_sub_batch_max_tokens.unwrap()),
            )
            .unwrap();
            let floor = profile.vulkan_min_device_local_bytes.unwrap();
            let arena = Layout::new(&plan, floor).unwrap();
            assert_eq!(arena.requested_bytes, floor);
            assert_eq!(floor % BUFFER_ALIGNMENT, 0);
            let mut peak = 0;
            for _sequence in 0..256 {
                // The actual runtime uses one sequence per sub-batch and retains this arena.
                peak = peak.max(arena.requested_bytes);
                for (offset, size) in arena.slices.values() {
                    assert_eq!(offset % BUFFER_ALIGNMENT, 0);
                    assert!(offset + size <= floor);
                }
            }
            assert_eq!(peak, floor);
            assert_eq!(
                Layout::new(&plan, floor - 1).unwrap_err(),
                "vulkan_insufficient_memory"
            );
        }
    }
}
