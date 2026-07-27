#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub struct ModernBertLayer<'a> {
    pub qkv_weight: &'a [f32],
    pub attention_output_weight: &'a [f32],
    pub attention_norm_weight: Option<&'a [f32]>,
    pub mlp_input_weight: &'a [f32],
    pub mlp_output_weight: &'a [f32],
    pub mlp_norm_weight: &'a [f32],
    pub sliding_attention: bool,
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub struct Qwen3Layer<'a> {
    pub input_norm: &'a [f32],
    pub post_attention_norm: &'a [f32],
    pub q_weight: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_weight: &'a [f32],
    pub k_norm: &'a [f32],
    pub v_weight: &'a [f32],
    pub o_weight: &'a [f32],
    pub gate_weight: &'a [f32],
    pub up_weight: &'a [f32],
    pub down_weight: &'a [f32],
}

#[cfg(feature = "vulkan")]
#[cfg_attr(target_os = "macos", allow(dead_code))]
mod enabled {
    use std::collections::HashMap;
    use std::ffi::{CStr, CString};
    use std::io::{Cursor, Write};
    use std::mem::size_of;
    use std::path::PathBuf;
    use std::ptr;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Once};
    use std::time::Instant;

    use anyhow::{ensure, Context, Result};
    use ash::{vk, Device, Entry, Instance};

    use super::super::{decode_f16_bits, encode_f16_bits, EncoderLayer, VulkanGemm};
    use super::{ModernBertLayer, Qwen3Layer};
    use crate::qwen3::{Model, Weight};

    const DESCRIPTOR_BINDINGS: u32 = 10;
    const PUSH_CONSTANT_BYTES: u32 = 128;
    static HOST_STORAGE_MEMORY_TYPE_REPORT: Once = Once::new();
    static STAGING_MEMORY_TYPE_REPORT: Once = Once::new();
    static WEIGHT_MEMORY_TYPE_REPORT: Once = Once::new();

    #[derive(Clone, Copy)]
    enum MemoryPool {
        HostStorage,
        Staging,
        Weight,
    }

    impl MemoryPool {
        fn label(self) -> &'static str {
            match self {
                Self::HostStorage => "host-storage",
                Self::Staging => "staging",
                Self::Weight => "weight",
            }
        }

        fn report(self) -> &'static Once {
            match self {
                Self::HostStorage => &HOST_STORAGE_MEMORY_TYPE_REPORT,
                Self::Staging => &STAGING_MEMORY_TYPE_REPORT,
                Self::Weight => &WEIGHT_MEMORY_TYPE_REPORT,
            }
        }
    }

    fn select_memory_type(
        memory_types: &[vk::MemoryType],
        bits: u32,
        required: vk::MemoryPropertyFlags,
        forbidden: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        memory_types
            .iter()
            .enumerate()
            .find(|(index, memory)| {
                bits & (1 << index) != 0
                    && memory.property_flags.contains(required)
                    && !memory.property_flags.intersects(forbidden)
            })
            .map(|(index, _)| index as u32)
    }

    #[cfg(test)]
    mod memory_tests {
        use super::*;

        #[test]
        fn weight_memory_excludes_host_visible_device_local_types() {
            let types = [
                vk::MemoryType::default().property_flags(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                ),
                vk::MemoryType::default().property_flags(vk::MemoryPropertyFlags::DEVICE_LOCAL),
            ];
            assert_eq!(
                select_memory_type(
                    &types,
                    0b11,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                ),
                Some(1)
            );
            assert_eq!(
                select_memory_type(
                    &types[..1],
                    0b1,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                ),
                None
            );
        }
    }

    struct Buffer {
        state: Arc<DeviceState>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        bytes: vk::DeviceSize,
    }

    impl Buffer {
        fn new(state: Arc<DeviceState>, bytes: usize) -> Result<Self> {
            Self::allocate(
                state,
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::empty(),
                MemoryPool::HostStorage,
            )
        }

        fn allocate(
            state: Arc<DeviceState>,
            bytes: usize,
            usage: vk::BufferUsageFlags,
            required: vk::MemoryPropertyFlags,
            forbidden: vk::MemoryPropertyFlags,
            pool: MemoryPool,
        ) -> Result<Self> {
            let bytes = bytes.max(4) as vk::DeviceSize;
            let create = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = unsafe { state.device.create_buffer(&create, None)? };
            let requirements = unsafe { state.device.get_buffer_memory_requirements(buffer) };
            let Some(memory_type) =
                state.memory_type(requirements.memory_type_bits, required, forbidden)
            else {
                unsafe { state.device.destroy_buffer(buffer, None) };
                anyhow::bail!(
                    "no Vulkan {} memory with required={required:?} forbidden={forbidden:?}",
                    pool.label()
                );
            };
            let memory_properties = state.memory_properties.memory_types[memory_type as usize];
            let heap_index = memory_properties.heap_index;
            if matches!(pool, MemoryPool::Weight) {
                if let Err(error) = state.ensure_heap_budget(heap_index, requirements.size) {
                    unsafe { state.device.destroy_buffer(buffer, None) };
                    return Err(error);
                }
            }
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            let memory = match unsafe { state.device.allocate_memory(&allocate, None) } {
                Ok(memory) => memory,
                Err(error) => {
                    unsafe { state.device.destroy_buffer(buffer, None) };
                    return Err(error.into());
                }
            };
            if let Err(error) = unsafe { state.device.bind_buffer_memory(buffer, memory, 0) } {
                unsafe {
                    state.device.free_memory(memory, None);
                    state.device.destroy_buffer(buffer, None);
                }
                return Err(error.into());
            }
            pool.report().call_once(|| {
                let budget = state.heap_budget(heap_index);
                eprintln!(
                    "Vulkan memory pool: pool={} type_index={memory_type} heap_index={heap_index} flags={:?} heap_size={} heap_usage={} heap_budget={}",
                    pool.label(),
                    memory_properties.property_flags,
                    state.memory_properties.memory_heaps[heap_index as usize].size,
                    budget.map_or(0, |value| value.0),
                    budget.map_or(0, |value| value.1),
                );
            });
            if matches!(pool, MemoryPool::Weight) {
                state
                    .immutable_allocation_count
                    .fetch_add(1, Ordering::Relaxed);
                state
                    .immutable_allocation_bytes
                    .fetch_add(requirements.size, Ordering::Relaxed);
            }
            Ok(Self {
                state,
                buffer,
                memory,
                bytes,
            })
        }

        fn staging(state: Arc<DeviceState>, bytes: usize) -> Result<Self> {
            Self::allocate(
                state,
                bytes,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::empty(),
                MemoryPool::Staging,
            )
        }

        fn from_f16(state: Arc<DeviceState>, values: &[f32]) -> Result<Self> {
            Self::from_slice(state, &encode_f16_bits(values))
        }

        fn from_f32(state: Arc<DeviceState>, values: &[f32]) -> Result<Self> {
            Self::from_slice(state, values)
        }

        fn from_slice<T: Copy>(state: Arc<DeviceState>, values: &[T]) -> Result<Self> {
            let bytes = std::mem::size_of_val(values);
            let destination = Self::allocate(
                state.clone(),
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                MemoryPool::Weight,
            )?;
            if bytes > 0 {
                let staging = Self::staging(state.clone(), bytes)?;
                staging.write(values)?;
                state.copy_buffer(staging.buffer, destination.buffer, bytes as u64)?;
            }
            Ok(destination)
        }

        fn write<T: Copy>(&self, values: &[T]) -> Result<()> {
            let bytes = std::mem::size_of_val(values) as vk::DeviceSize;
            ensure!(
                bytes <= self.bytes,
                "Vulkan buffer write exceeds allocation"
            );
            unsafe {
                let mapped = self.state.device.map_memory(
                    self.memory,
                    0,
                    bytes,
                    vk::MemoryMapFlags::empty(),
                )?;
                ptr::copy_nonoverlapping(
                    values.as_ptr().cast::<u8>(),
                    mapped.cast(),
                    bytes as usize,
                );
                self.state.device.unmap_memory(self.memory);
            }
            Ok(())
        }

        fn read_u16(&self, count: usize) -> Result<Vec<u16>> {
            ensure!(
                count * size_of::<u16>() <= self.bytes as usize,
                "Vulkan buffer read exceeds allocation"
            );
            let mut values = vec![0u16; count];
            unsafe {
                let mapped = self.state.device.map_memory(
                    self.memory,
                    0,
                    (count * size_of::<u16>()) as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                ptr::copy_nonoverlapping(mapped.cast::<u16>(), values.as_mut_ptr(), count);
                self.state.device.unmap_memory(self.memory);
            }
            Ok(values)
        }

        fn read_f32(&self, count: usize) -> Result<Vec<f32>> {
            ensure!(
                count * size_of::<f32>() <= self.bytes as usize,
                "Vulkan buffer read exceeds allocation"
            );
            let mut values = vec![0.0f32; count];
            unsafe {
                let mapped = self.state.device.map_memory(
                    self.memory,
                    0,
                    (count * size_of::<f32>()) as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                ptr::copy_nonoverlapping(mapped.cast::<f32>(), values.as_mut_ptr(), count);
                self.state.device.unmap_memory(self.memory);
            }
            Ok(values)
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            unsafe {
                self.state.device.destroy_buffer(self.buffer, None);
                self.state.device.free_memory(self.memory, None);
            }
        }
    }

    struct DeviceState {
        _entry: Entry,
        instance: Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        queue: vk::Queue,
        queue_family: u32,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        memory_budget_supported: bool,
        upload_command_pool: vk::CommandPool,
        upload_fence: vk::Fence,
        immutable_allocation_count: AtomicUsize,
        immutable_allocation_bytes: AtomicU64,
        descriptor_layout: vk::DescriptorSetLayout,
        pipeline_layout: vk::PipelineLayout,
        pipeline_cache: vk::PipelineCache,
        pipeline_cache_path: Option<PathBuf>,
        profile_enabled: bool,
        profile_output: Option<PathBuf>,
        timestamp_period_ns: f64,
        timestamp_valid_bits: u32,
        gemm: VulkanGemm,
        shader_int8_supported: bool,
        subgroup_size: u32,
    }

    impl DeviceState {
        fn new(gemm: VulkanGemm, pipeline_cache_path: Option<PathBuf>) -> Result<Arc<Self>> {
            unsafe {
                let started = Instant::now();
                let entry = Entry::load().context("load Vulkan loader")?;
                eprintln!(
                    "Vulkan phase: loader_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1_000.0
                );
                let app_name = CString::new("synapse-vulkan-minilm")?;
                let app = vk::ApplicationInfo::default()
                    .application_name(&app_name)
                    .api_version(vk::API_VERSION_1_3);
                let instance_started = Instant::now();
                let instance = entry.create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )?;
                eprintln!(
                    "Vulkan phase: instance_ms={:.3}",
                    instance_started.elapsed().as_secs_f64() * 1_000.0
                );
                let physical_device = instance
                    .enumerate_physical_devices()?
                    .into_iter()
                    .find(|physical| {
                        let properties = instance.get_physical_device_properties(*physical);
                        matches!(
                            properties.device_type,
                            vk::PhysicalDeviceType::INTEGRATED_GPU
                                | vk::PhysicalDeviceType::DISCRETE_GPU
                        )
                    })
                    .context("no Vulkan compute GPU found")?;
                let properties = instance.get_physical_device_properties(physical_device);
                let device_name = CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy();
                let subgroup = {
                    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
                    let mut properties2 =
                        vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
                    instance.get_physical_device_properties2(physical_device, &mut properties2);
                    subgroup
                };
                let queue_families =
                    instance.get_physical_device_queue_family_properties(physical_device);
                let queue_family = queue_families
                    .iter()
                    .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .context("Vulkan GPU has no compute queue")?
                    as u32;
                let profile_requested = std::env::var_os("SYNAPSE_VULKAN_PROFILE").is_some()
                    || std::env::var_os("SYNAPSE_VULKAN_PROFILE_OUT").is_some();
                let timestamp_period_ns = f64::from(properties.limits.timestamp_period);
                let timestamp_valid_bits =
                    queue_families[queue_family as usize].timestamp_valid_bits;
                if profile_requested {
                    ensure!(
                        properties.limits.timestamp_compute_and_graphics != 0,
                        "Vulkan timestamp profiling requested, but timestampComputeAndGraphics is false"
                    );
                    ensure!(
                        timestamp_period_ns.is_finite() && timestamp_period_ns > 0.0,
                        "Vulkan timestamp profiling requested, but timestampPeriod is invalid: {}",
                        properties.limits.timestamp_period
                    );
                    ensure!(
                        timestamp_valid_bits > 0,
                        "Vulkan timestamp profiling requested, but the compute queue exposes no timestamp bits"
                    );
                    eprintln!(
                        "Vulkan timestamps: timestampComputeAndGraphics=true timestampPeriod_ns={timestamp_period_ns:.6} compute_queue_valid_bits={timestamp_valid_bits}"
                    );
                }
                let profile_output = profile_requested
                    .then(|| std::env::var_os("SYNAPSE_VULKAN_PROFILE_OUT").map(PathBuf::from))
                    .flatten();
                let memory_budget_supported = instance
                    .enumerate_device_extension_properties(physical_device)?
                    .iter()
                    .any(|extension| {
                        CStr::from_ptr(extension.extension_name.as_ptr())
                            == ash::ext::memory_budget::NAME
                    });
                let memory_properties =
                    instance.get_physical_device_memory_properties(physical_device);
                let budget_snapshot = memory_budget_supported.then(|| {
                    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
                    let mut properties =
                        vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
                    instance
                        .get_physical_device_memory_properties2(physical_device, &mut properties);
                    budget
                });
                for heap_index in 0..memory_properties.memory_heap_count as usize {
                    let heap = memory_properties.memory_heaps[heap_index];
                    eprintln!(
                        "Vulkan memory heap: heap_index={heap_index} size={} flags={:?} usage={} budget={} memory_budget_ext={memory_budget_supported}",
                        heap.size,
                        heap.flags,
                        budget_snapshot.as_ref().map_or(0, |budget| budget.heap_usage[heap_index]),
                        budget_snapshot.as_ref().map_or(0, |budget| budget.heap_budget[heap_index]),
                    );
                }

                let mut supported_storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
                let mut supported_float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
                let mut supported_scalar = vk::PhysicalDeviceScalarBlockLayoutFeatures::default();
                let mut supported_cooperative =
                    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
                let mut features = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut supported_storage16)
                    .push_next(&mut supported_float16)
                    .push_next(&mut supported_scalar)
                    .push_next(&mut supported_cooperative);
                instance.get_physical_device_features2(physical_device, &mut features);
                ensure!(
                    supported_storage16.storage_buffer16_bit_access != 0,
                    "Vulkan GPU lacks 16-bit storage buffers"
                );
                ensure!(
                    supported_float16.shader_float16 != 0,
                    "Vulkan GPU lacks shader float16"
                );
                ensure!(
                    supported_scalar.scalar_block_layout != 0,
                    "Vulkan GPU lacks scalar block layout"
                );
                if matches!(gemm, VulkanGemm::Cooperative) {
                    ensure!(
                        supported_cooperative.cooperative_matrix != 0,
                        "Vulkan GPU lacks cooperative matrices"
                    );
                    ensure!(
                        subgroup.subgroup_size == 64,
                        "cooperative shader requires the queried RDNA3 subgroup size 64, got {}",
                        subgroup.subgroup_size
                    );
                    let matrix_loader =
                        ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
                    let supported = matrix_loader
                        .get_physical_device_cooperative_matrix_properties(physical_device)?
                        .into_iter()
                        .any(|property| {
                            property.m_size == 16
                                && property.n_size == 16
                                && property.k_size == 16
                                && property.a_type == vk::ComponentTypeKHR::FLOAT16
                                && property.b_type == vk::ComponentTypeKHR::FLOAT16
                                && property.c_type == vk::ComponentTypeKHR::FLOAT32
                                && property.result_type == vk::ComponentTypeKHR::FLOAT32
                        });
                    ensure!(
                        supported,
                        "Vulkan GPU lacks 16x16x16 f16/f16/f32 cooperative matrices"
                    );
                }

                let priority = [1.0f32];
                let queue = [vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family)
                    .queue_priorities(&priority)];
                let mut extension_names = Vec::new();
                if matches!(gemm, VulkanGemm::Cooperative) {
                    extension_names.push(ash::khr::cooperative_matrix::NAME.as_ptr());
                }
                if memory_budget_supported {
                    extension_names.push(ash::ext::memory_budget::NAME.as_ptr());
                }
                let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default()
                    .storage_buffer16_bit_access(true);
                ensure!(
                    subgroup.subgroup_size > 0 && 64 % subgroup.subgroup_size == 0,
                    "Vulkan decode requires a subgroup size that divides the 64-invocation workgroup, got {}",
                    subgroup.subgroup_size
                );
                let shader_int8_supported = supported_float16.shader_int8 != 0;
                let mut float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
                    .shader_float16(true)
                    .shader_int8(shader_int8_supported);
                let mut scalar = vk::PhysicalDeviceScalarBlockLayoutFeatures::default()
                    .scalar_block_layout(true);
                let mut cooperative = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
                    .cooperative_matrix(matches!(gemm, VulkanGemm::Cooperative));
                let create = vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue)
                    .enabled_extension_names(&extension_names)
                    .push_next(&mut storage16)
                    .push_next(&mut float16)
                    .push_next(&mut scalar)
                    .push_next(&mut cooperative);
                let device_started = Instant::now();
                let device = instance.create_device(physical_device, &create, None)?;
                let queue = device.get_device_queue(queue_family, 0);
                let upload_command_pool = device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )?;
                let upload_fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
                eprintln!(
                    "Vulkan phase: device_ms={:.3}",
                    device_started.elapsed().as_secs_f64() * 1_000.0
                );

                let bindings = (0..DESCRIPTOR_BINDINGS)
                    .map(|binding| {
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    })
                    .collect::<Vec<_>>();
                let descriptor_layout = device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )?;
                let set_layouts = [descriptor_layout];
                let push_range = [vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .size(PUSH_CONSTANT_BYTES)];
                let pipeline_layout = device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_range),
                    None,
                )?;
                let initial_cache = pipeline_cache_path
                    .as_ref()
                    .and_then(|path| std::fs::read(path).ok())
                    .unwrap_or_default();
                let pipeline_cache = device.create_pipeline_cache(
                    &vk::PipelineCacheCreateInfo::default().initial_data(&initial_cache),
                    None,
                )?;
                eprintln!(
                    "Vulkan init: device={device_name} api={}.{}.{} driver_raw={} subgroup={} gemm={} pipeline_cache_input_bytes={}",
                    vk::api_version_major(properties.api_version),
                    vk::api_version_minor(properties.api_version),
                    vk::api_version_patch(properties.api_version),
                    properties.driver_version,
                    subgroup.subgroup_size,
                    gemm.as_str(),
                    initial_cache.len()
                );
                Ok(Arc::new(Self {
                    _entry: entry,
                    instance,
                    physical_device,
                    device,
                    queue,
                    queue_family,
                    memory_properties,
                    memory_budget_supported,
                    upload_command_pool,
                    upload_fence,
                    immutable_allocation_count: AtomicUsize::new(0),
                    immutable_allocation_bytes: AtomicU64::new(0),
                    descriptor_layout,
                    pipeline_layout,
                    pipeline_cache,
                    pipeline_cache_path,
                    profile_enabled: profile_requested,
                    profile_output,
                    timestamp_period_ns,
                    timestamp_valid_bits,
                    gemm,
                    shader_int8_supported,
                    subgroup_size: subgroup.subgroup_size,
                }))
            }
        }

        // Decode shaders use a fixed 64-invocation workgroup so one binary
        // handles both RDNA3 wave32 and wave64. This computes the workgroups
        // needed when each subgroup owns a fixed number of independent rows.
        fn subgroup_groups(&self, rows: usize, rows_per_subgroup: u32) -> u32 {
            let subgroups_per_workgroup = 64 / self.subgroup_size;
            rows.div_ceil((subgroups_per_workgroup * rows_per_subgroup) as usize) as u32
        }

        fn memory_type(
            &self,
            bits: u32,
            required: vk::MemoryPropertyFlags,
            forbidden: vk::MemoryPropertyFlags,
        ) -> Option<u32> {
            select_memory_type(
                &self.memory_properties.memory_types
                    [..self.memory_properties.memory_type_count as usize],
                bits,
                required,
                forbidden,
            )
        }

        fn heap_budget(&self, heap_index: u32) -> Option<(vk::DeviceSize, vk::DeviceSize)> {
            self.memory_budget_supported.then(|| unsafe {
                let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
                let mut properties =
                    vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
                self.instance
                    .get_physical_device_memory_properties2(self.physical_device, &mut properties);
                (
                    budget.heap_usage[heap_index as usize],
                    budget.heap_budget[heap_index as usize],
                )
            })
        }

        fn ensure_heap_budget(&self, heap_index: u32, allocation: vk::DeviceSize) -> Result<()> {
            if let Some((usage, budget)) = self.heap_budget(heap_index) {
                ensure!(
                    usage.saturating_add(allocation) <= budget,
                    "Vulkan device-local heap {heap_index} budget exhausted: usage={usage} allocation={allocation} budget={budget}"
                );
            }
            Ok(())
        }

        fn copy_buffer(
            &self,
            source: vk::Buffer,
            destination: vk::Buffer,
            bytes: vk::DeviceSize,
        ) -> Result<()> {
            let command = unsafe {
                self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.upload_command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0]
            };
            let result = (|| unsafe {
                self.device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                self.device.cmd_copy_buffer(
                    command,
                    source,
                    destination,
                    &[vk::BufferCopy::default().size(bytes)],
                );
                self.device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .buffer(destination)
                        .size(bytes)],
                    &[],
                );
                self.device.end_command_buffer(command)?;
                self.device.reset_fences(&[self.upload_fence])?;
                self.device.queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command])],
                    self.upload_fence,
                )?;
                self.device
                    .wait_for_fences(&[self.upload_fence], true, u64::MAX)?;
                Ok::<_, anyhow::Error>(())
            })();
            unsafe {
                self.device
                    .free_command_buffers(self.upload_command_pool, &[command]);
            }
            result
        }

        fn immutable_allocation_summary(&self) -> (usize, u64) {
            (
                self.immutable_allocation_count.load(Ordering::Relaxed),
                self.immutable_allocation_bytes.load(Ordering::Relaxed),
            )
        }
    }

    impl Drop for DeviceState {
        fn drop(&mut self) {
            unsafe {
                let _ = self.device.device_wait_idle();
                if let Some(path) = &self.pipeline_cache_path {
                    match self.device.get_pipeline_cache_data(self.pipeline_cache) {
                        Ok(bytes) => {
                            if let Some(parent) = path.parent() {
                                let _ = std::fs::create_dir_all(parent);
                            }
                            if let Err(error) = std::fs::write(path, &bytes) {
                                eprintln!("Vulkan pipeline cache write failed: {error}");
                            } else {
                                eprintln!(
                                    "Vulkan pipeline cache output: bytes={} path={}",
                                    bytes.len(),
                                    path.display()
                                );
                            }
                        }
                        Err(error) => {
                            eprintln!("Vulkan pipeline cache retrieval failed: {error:?}")
                        }
                    }
                }
                self.device
                    .destroy_pipeline_cache(self.pipeline_cache, None);
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_layout, None);
                self.device.destroy_fence(self.upload_fence, None);
                self.device
                    .destroy_command_pool(self.upload_command_pool, None);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    struct DeviceLinear {
        weight: Buffer,
        bias: Buffer,
    }

    struct DeviceLayer {
        query: DeviceLinear,
        key: DeviceLinear,
        value: DeviceLinear,
        attention_output: DeviceLinear,
        attention_ln_weight: Buffer,
        attention_ln_bias: Buffer,
        intermediate: DeviceLinear,
        output: DeviceLinear,
        output_ln_weight: Buffer,
        output_ln_bias: Buffer,
    }

    impl DeviceLinear {
        fn upload(state: Arc<DeviceState>, linear: &super::super::Linear) -> Result<Self> {
            Ok(Self {
                weight: Buffer::from_f16(state.clone(), &linear.weight.data)?,
                bias: Buffer::from_f16(state, linear.bias.as_slice())?,
            })
        }
    }

    impl DeviceLayer {
        fn upload(state: Arc<DeviceState>, layer: &EncoderLayer) -> Result<Self> {
            Ok(Self {
                query: DeviceLinear::upload(state.clone(), &layer.query)?,
                key: DeviceLinear::upload(state.clone(), &layer.key)?,
                value: DeviceLinear::upload(state.clone(), &layer.value)?,
                attention_output: DeviceLinear::upload(state.clone(), &layer.attention_output)?,
                attention_ln_weight: Buffer::from_f32(
                    state.clone(),
                    layer.attention_ln_weight.as_slice(),
                )?,
                attention_ln_bias: Buffer::from_f32(
                    state.clone(),
                    layer.attention_ln_bias.as_slice(),
                )?,
                intermediate: DeviceLinear::upload(state.clone(), &layer.intermediate)?,
                output: DeviceLinear::upload(state.clone(), &layer.output)?,
                output_ln_weight: Buffer::from_f32(
                    state.clone(),
                    layer.output_ln_weight.as_slice(),
                )?,
                output_ln_bias: Buffer::from_f32(state, layer.output_ln_bias.as_slice())?,
            })
        }
    }

    /// Wave-5 batched mat-mat pipelines. The four mat-mat shaders (f16 and Q8)
    /// are specialized per K in {1,2,4,8,16} so the K accumulators stay
    /// register-resident; the pointwise batched shaders (RMSNorm, head-norm+
    /// RoPE, value-cache, attention, add_residual, swiglu) read K from a push
    /// constant and are shared across K. `decode_matvec_q8_0_batch` is only
    /// created when the device supports shader int8.
    struct BatchedPipelines {
        matvec_f16: [vk::Pipeline; 5],          // K in {1,2,4,8,16}
        matvec_q8_0: Option<[vk::Pipeline; 5]>, // K in {1,2,4,8,16}
        /// Column-offset Q8 matvec for the batched fallback (K sequential
        /// single-token dispatches). Only created when shader int8 is
        /// supported.
        matvec_q8_0_column: Option<vk::Pipeline>,
        rms_norm: vk::Pipeline,
        head_norm_rope: vk::Pipeline,
        value_cache: vk::Pipeline,
        attention: vk::Pipeline,
        add_residual: vk::Pipeline,
        swiglu: vk::Pipeline,
    }

    impl BatchedPipelines {
        fn create(state: &DeviceState) -> Result<Self> {
            // Specialize the mat-mat shaders for K in {1,2,4,8,16}. The index
            // maps to the power-of-two column count: 0->1, 1->2, 2->4, 3->8,
            // 4->16. The host rounds the requested batch up to the next power
            // of two and dispatches only the first `batch` columns.
            let ks = [1u32, 2, 4, 8, 16];
            let matvec_f16 = ks
                .iter()
                .map(|&k| {
                    create_pipeline_specialized(
                        state,
                        include_bytes!("vulkan_spv/decode_matvec_batch.spv"),
                        k,
                    )
                })
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| anyhow::anyhow!("expected 5 f16 batched pipelines"))?;
            let matvec_q8_0 = if state.shader_int8_supported {
                let pipelines = ks
                    .iter()
                    .map(|&k| {
                        create_pipeline_specialized(
                            state,
                            include_bytes!("vulkan_spv/decode_matvec_q8_0_batch.spv"),
                            k,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("expected 5 q8 batched pipelines"))?;
                Some(pipelines)
            } else {
                None
            };
            let matvec_q8_0_column = state
                .shader_int8_supported
                .then(|| {
                    create_pipeline(
                        state,
                        include_bytes!("vulkan_spv/decode_matvec_q8_0_column.spv"),
                    )
                })
                .transpose()?;
            Ok(Self {
                matvec_f16,
                matvec_q8_0,
                matvec_q8_0_column,
                rms_norm: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/decode_rms_norm_batch.spv"),
                )?,
                head_norm_rope: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/decode_head_norm_rope_batch.spv"),
                )?,
                value_cache: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/decode_value_cache_batch.spv"),
                )?,
                attention: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/decode_attention_batch.spv"),
                )?,
                add_residual: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/add_residual_batch.spv"),
                )?,
                swiglu: create_pipeline(state, include_bytes!("vulkan_spv/swiglu_batch.spv"))?,
            })
        }

        fn all(&self) -> impl Iterator<Item = vk::Pipeline> + '_ {
            self.matvec_f16
                .iter()
                .copied()
                .chain(
                    self.matvec_q8_0
                        .iter()
                        .flat_map(|pipelines| pipelines.iter().copied()),
                )
                .chain(self.matvec_q8_0_column)
                .chain([
                    self.rms_norm,
                    self.head_norm_rope,
                    self.value_cache,
                    self.attention,
                    self.add_residual,
                    self.swiglu,
                ])
        }
    }

    struct Pipelines {
        plain: vk::Pipeline,
        cooperative: Option<vk::Pipeline>,
        qkv: vk::Pipeline,
        softmax: vk::Pipeline,
        transpose: vk::Pipeline,
        residual_norm: vk::Pipeline,
        gelu: vk::Pipeline,
        pool: vk::Pipeline,
        layer_norm: vk::Pipeline,
        rms_norm: vk::Pipeline,
        add_residual: vk::Pipeline,
        modern_qkv_rope: vk::Pipeline,
        modern_softmax: vk::Pipeline,
        geglu: vk::Pipeline,
        qwen_head_norm_rope: vk::Pipeline,
        qwen_value_transpose: vk::Pipeline,
        qwen_causal_softmax: vk::Pipeline,
        qwen_context_transpose: vk::Pipeline,
        swiglu: vk::Pipeline,
        decode_matvec: vk::Pipeline,
        decode_matvec_q8_0: Option<vk::Pipeline>,
        decode_rms_norm: vk::Pipeline,
        decode_head_norm_rope: vk::Pipeline,
        decode_value_cache: vk::Pipeline,
        decode_attention: vk::Pipeline,
        batched: BatchedPipelines,
    }

    impl Pipelines {
        fn create(state: &DeviceState) -> Result<Self> {
            eprintln!(
                "Vulkan phase: begin pipeline creation gemm={}",
                state.gemm.as_str()
            );
            Ok(Self {
                plain: create_pipeline(state, include_bytes!("vulkan_spv/gemm_plain.spv"))?,
                cooperative: matches!(state.gemm, VulkanGemm::Cooperative)
                    .then(|| {
                        create_pipeline(state, include_bytes!("vulkan_spv/gemm_cooperative.spv"))
                    })
                    .transpose()?,
                qkv: create_pipeline(state, include_bytes!("vulkan_spv/qkv_bias_transpose.spv"))?,
                softmax: create_pipeline(state, include_bytes!("vulkan_spv/softmax.spv"))?,
                transpose: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/transpose_context.spv"),
                )?,
                residual_norm: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/residual_layer_norm.spv"),
                )?,
                gelu: create_pipeline(state, include_bytes!("vulkan_spv/bias_gelu.spv"))?,
                pool: create_pipeline(state, include_bytes!("vulkan_spv/mean_pool_l2.spv"))?,
                layer_norm: create_pipeline(state, include_bytes!("vulkan_spv/layer_norm.spv"))?,
                rms_norm: create_pipeline(state, include_bytes!("vulkan_spv/rms_norm.spv"))?,
                add_residual: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/add_residual.spv"),
                )?,
                modern_qkv_rope: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/modern_qkv_rope.spv"),
                )?,
                modern_softmax: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/modern_softmax.spv"),
                )?,
                geglu: create_pipeline(state, include_bytes!("vulkan_spv/geglu.spv"))?,
                qwen_head_norm_rope: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/qwen_head_norm_rope.spv"),
                )?,
                qwen_value_transpose: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/qwen_value_transpose.spv"),
                )?,
                qwen_causal_softmax: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/qwen_causal_softmax.spv"),
                )?,
                qwen_context_transpose: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/qwen_context_transpose.spv"),
                )?,
                swiglu: create_pipeline(state, include_bytes!("vulkan_spv/swiglu.spv"))?,
                decode_matvec: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/decode_matvec.spv"),
                )?,
                decode_matvec_q8_0: state
                    .shader_int8_supported
                    .then(|| {
                        create_pipeline(state, include_bytes!("vulkan_spv/decode_matvec_q8_0.spv"))
                    })
                    .transpose()?,
                decode_rms_norm: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/decode_rms_norm.spv"),
                )?,
                decode_head_norm_rope: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/decode_head_norm_rope.spv"),
                )?,
                decode_value_cache: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/decode_value_cache.spv"),
                )?,
                decode_attention: create_pipeline(
                    state,
                    include_bytes!("vulkan_spv/decode_attention.spv"),
                )?,
                batched: BatchedPipelines::create(state)?,
            })
        }

        fn all(&self) -> impl Iterator<Item = vk::Pipeline> + '_ {
            [
                Some(self.plain),
                self.cooperative,
                Some(self.qkv),
                Some(self.softmax),
                Some(self.transpose),
                Some(self.residual_norm),
                Some(self.gelu),
                Some(self.pool),
                Some(self.layer_norm),
                Some(self.rms_norm),
                Some(self.add_residual),
                Some(self.modern_qkv_rope),
                Some(self.modern_softmax),
                Some(self.geglu),
                Some(self.qwen_head_norm_rope),
                Some(self.qwen_value_transpose),
                Some(self.qwen_causal_softmax),
                Some(self.qwen_context_transpose),
                Some(self.swiglu),
                Some(self.decode_matvec),
                self.decode_matvec_q8_0,
                Some(self.decode_rms_norm),
                Some(self.decode_head_norm_rope),
                Some(self.decode_value_cache),
                Some(self.decode_attention),
            ]
            .into_iter()
            .flatten()
            .chain(self.batched.all())
        }
    }

    fn create_pipeline(state: &DeviceState, spirv: &[u8]) -> Result<vk::Pipeline> {
        let words = ash::util::read_spv(&mut Cursor::new(spirv))?;
        let module = unsafe {
            state
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
        };
        let main = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(main);
        let create = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(state.pipeline_layout);
        let pipeline = unsafe {
            state
                .device
                .create_compute_pipelines(state.pipeline_cache, &[create], None)
                .map_err(|(_, error)| error)?[0]
        };
        unsafe { state.device.destroy_shader_module(module, None) };
        Ok(pipeline)
    }

    /// Create a compute pipeline with a single u32 specialization constant at
    /// SpecId 0. The wave-5 batched mat-mat shaders specialize the column count
    /// K so the K accumulators stay register-resident and each column's
    /// accumulation keeps the exact ascending order of the K=1 path. A runtime
    /// column count would spill the accumulators to memory and let the compiler
    /// fold products before adding them to the running sum, which reorders the
    /// f32 rounding and breaks the byte-exact gate.
    fn create_pipeline_specialized(
        state: &DeviceState,
        spirv: &[u8],
        constant_value: u32,
    ) -> Result<vk::Pipeline> {
        let words = ash::util::read_spv(&mut Cursor::new(spirv))?;
        let module = unsafe {
            state
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
        };
        let main = c"main";
        let spec_data = constant_value.to_ne_bytes();
        let map_entry = vk::SpecializationMapEntry::default()
            .constant_id(0)
            .size(size_of::<u32>())
            .offset(0);
        let map_entries = [map_entry];
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&map_entries)
            .data(&spec_data);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(main)
            .specialization_info(&specialization);
        let create = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(state.pipeline_layout);
        let pipeline = unsafe {
            state
                .device
                .create_compute_pipelines(state.pipeline_cache, &[create], None)
                .map_err(|(_, error)| error)?[0]
        };
        unsafe { state.device.destroy_shader_module(module, None) };
        Ok(pipeline)
    }

    struct Activations {
        input: Buffer,
        mask: Buffer,
        x1: Buffer,
        q_raw: Buffer,
        k_raw: Buffer,
        v_raw: Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        scores_f32: Buffer,
        scores_f16: Buffer,
        attention_f32: Buffer,
        context_f16: Buffer,
        projected_f32: Buffer,
        intermediate_f32: Buffer,
        intermediate_f16: Buffer,
        ffn_f32: Buffer,
        pooled: Buffer,
    }

    impl Activations {
        fn new(
            state: Arc<DeviceState>,
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
        ) -> Result<Self> {
            let rows = batch * seq;
            let hidden_values = rows * hidden;
            let score_values = batch * heads * seq * seq;
            let intermediate_values = rows * intermediate;
            let f16 = |count| Buffer::new(state.clone(), count * size_of::<u16>());
            let f32_buffer = |count| Buffer::new(state.clone(), count * size_of::<f32>());
            Ok(Self {
                input: f16(hidden_values)?,
                mask: Buffer::new(state.clone(), rows * size_of::<u32>())?,
                x1: f16(hidden_values)?,
                q_raw: f32_buffer(hidden_values)?,
                k_raw: f32_buffer(hidden_values)?,
                v_raw: f32_buffer(hidden_values)?,
                q: f16(hidden_values)?,
                k: f16(hidden_values)?,
                v: f16(hidden_values)?,
                scores_f32: f32_buffer(score_values)?,
                scores_f16: f16(score_values)?,
                attention_f32: f32_buffer(hidden_values)?,
                context_f16: f16(hidden_values)?,
                projected_f32: f32_buffer(hidden_values)?,
                intermediate_f32: f32_buffer(intermediate_values)?,
                intermediate_f16: f16(intermediate_values)?,
                ffn_f32: f32_buffer(hidden_values)?,
                pooled: f32_buffer(batch * hidden)?,
            })
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct GemmParams {
        m: u32,
        n: u32,
        k: u32,
        batch_count: u32,
        transpose_b: u32,
        edge_only: u32,
        a_offset: u32,
        b_offset: u32,
        c_offset: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct FourU32 {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct SoftmaxParams {
        batch: u32,
        heads: u32,
        seq: u32,
        scale: f32,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct NormParams {
        rows: u32,
        hidden: u32,
        epsilon: f32,
    }

    const PROFILE_STAGE_COUNT: usize = 10;

    #[derive(Copy, Clone)]
    #[repr(usize)]
    enum StageClass {
        GemmQkv,
        GemmAttentionScores,
        SoftmaxMask,
        GemmPv,
        GemmOut,
        GemmMlpUp,
        GemmMlpDown,
        Pointwise,
        LayoutTranspose,
        Readback,
    }

    impl StageClass {
        const ALL: [Self; PROFILE_STAGE_COUNT] = [
            Self::GemmQkv,
            Self::GemmAttentionScores,
            Self::SoftmaxMask,
            Self::GemmPv,
            Self::GemmOut,
            Self::GemmMlpUp,
            Self::GemmMlpDown,
            Self::Pointwise,
            Self::LayoutTranspose,
            Self::Readback,
        ];

        fn label(self) -> &'static str {
            match self {
                Self::GemmQkv => "GEMM-qkv",
                Self::GemmAttentionScores => "GEMM-attn-scores",
                Self::SoftmaxMask => "softmax+mask",
                Self::GemmPv => "GEMM-PV",
                Self::GemmOut => "GEMM-out",
                Self::GemmMlpUp => "GEMM-mlp-up",
                Self::GemmMlpDown => "GEMM-mlp-down",
                Self::Pointwise => "pointwise",
                Self::LayoutTranspose => "layout/transpose",
                Self::Readback => "readback",
            }
        }
    }

    struct DispatchQueries {
        layer: usize,
        stage: StageClass,
        start: u32,
        dispatch_end: u32,
        barrier_end: u32,
    }

    struct QueryRecording {
        pool: vk::QueryPool,
        capacity: u32,
        next_query: u32,
        dispatches: Vec<DispatchQueries>,
        total_end: Option<u32>,
        empty_start: Option<u32>,
        empty_end: Option<u32>,
    }

    impl QueryRecording {
        fn new(
            state: &DeviceState,
            command: vk::CommandBuffer,
            max_dispatches: u32,
        ) -> Result<Option<Self>> {
            if !state.profile_enabled {
                return Ok(None);
            }
            let capacity = max_dispatches * 3 + 4;
            let pool = unsafe {
                state.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(capacity),
                    None,
                )?
            };
            unsafe {
                state
                    .device
                    .cmd_reset_query_pool(command, pool, 0, capacity);
                state.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    pool,
                    0,
                );
            }
            Ok(Some(Self {
                pool,
                capacity,
                next_query: 1,
                dispatches: Vec::new(),
                total_end: None,
                empty_start: None,
                empty_end: None,
            }))
        }

        fn reserve_dispatch(&mut self, layer: usize, stage: StageClass) -> Result<(u32, u32, u32)> {
            ensure!(
                self.next_query + 3 <= self.capacity - 3,
                "Vulkan profile query pool is too small"
            );
            let queries = (self.next_query, self.next_query + 1, self.next_query + 2);
            self.next_query += 3;
            self.dispatches.push(DispatchQueries {
                layer,
                stage,
                start: queries.0,
                dispatch_end: queries.1,
                barrier_end: queries.2,
            });
            Ok(queries)
        }

        fn finish(&mut self, state: &DeviceState, command: vk::CommandBuffer) -> Result<()> {
            ensure!(
                self.next_query + 3 <= self.capacity,
                "Vulkan profile query pool cannot fit terminal timestamps"
            );
            let total_end = self.next_query;
            let empty_start = total_end + 1;
            let empty_end = total_end + 2;
            unsafe {
                state.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.pool,
                    total_end,
                );
                // Adjacent timestamps quantify query-command overhead without removing a data
                // dependency that the encoder requires for correctness.
                state.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.pool,
                    empty_start,
                );
                state.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.pool,
                    empty_end,
                );
            }
            self.next_query += 3;
            self.total_end = Some(total_end);
            self.empty_start = Some(empty_start);
            self.empty_end = Some(empty_end);
            Ok(())
        }
    }

    #[derive(Default)]
    struct LayerProfile {
        stage_us: [f64; PROFILE_STAGE_COUNT],
        barrier_us: f64,
    }

    struct ProfileCapture {
        query: QueryRecording,
        family: &'static str,
        batch: usize,
        seq: usize,
        layer_count: usize,
        samples_seen: u64,
        recording_current_sample: bool,
        executions: u64,
        layers: Vec<LayerProfile>,
        gpu_total_us: f64,
        gpu_interval_residual_us: f64,
        submit_wait_us: f64,
        empty_sandwich_us: f64,
    }

    impl ProfileCapture {
        fn new(
            query: QueryRecording,
            family: &'static str,
            batch: usize,
            seq: usize,
            layer_count: usize,
        ) -> Self {
            Self {
                query,
                family,
                batch,
                seq,
                layer_count,
                samples_seen: 0,
                recording_current_sample: false,
                executions: 0,
                layers: (0..=layer_count).map(|_| LayerProfile::default()).collect(),
                gpu_total_us: 0.0,
                gpu_interval_residual_us: 0.0,
                submit_wait_us: 0.0,
                empty_sandwich_us: 0.0,
            }
        }

        fn ticks_between(state: &DeviceState, start: u64, end: u64) -> u64 {
            let delta = end.wrapping_sub(start);
            if state.timestamp_valid_bits >= 64 {
                delta
            } else {
                delta & ((1_u64 << state.timestamp_valid_bits) - 1)
            }
        }

        fn microseconds(state: &DeviceState, start: u64, end: u64) -> f64 {
            Self::ticks_between(state, start, end) as f64 * state.timestamp_period_ns / 1_000.0
        }

        fn record_queries(&mut self, state: &DeviceState, submit_wait_us: f64) -> Result<()> {
            let mut timestamps = vec![0_u64; self.query.next_query as usize];
            unsafe {
                state.device.get_query_pool_results(
                    self.query.pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
            }
            self.samples_seen += 1;
            if self.samples_seen == 1 {
                self.recording_current_sample = false;
                return Ok(());
            }
            self.recording_current_sample = true;
            let total_end = self
                .query
                .total_end
                .context("profile total timestamp missing")?;
            let empty_start = self
                .query
                .empty_start
                .context("profile empty-sandwich start missing")?;
            let empty_end = self
                .query
                .empty_end
                .context("profile empty-sandwich end missing")?;
            let total_us = Self::microseconds(state, timestamps[0], timestamps[total_end as usize]);
            let mut attributed_us = 0.0;
            for dispatch in &self.query.dispatches {
                let dispatch_us = Self::microseconds(
                    state,
                    timestamps[dispatch.start as usize],
                    timestamps[dispatch.dispatch_end as usize],
                );
                let barrier_us = Self::microseconds(
                    state,
                    timestamps[dispatch.dispatch_end as usize],
                    timestamps[dispatch.barrier_end as usize],
                );
                self.layers[dispatch.layer].stage_us[dispatch.stage as usize] += dispatch_us;
                self.layers[dispatch.layer].barrier_us += barrier_us;
                attributed_us += dispatch_us + barrier_us;
            }
            self.executions += 1;
            self.gpu_total_us += total_us;
            self.gpu_interval_residual_us += total_us - attributed_us;
            self.submit_wait_us += submit_wait_us;
            self.empty_sandwich_us += Self::microseconds(
                state,
                timestamps[empty_start as usize],
                timestamps[empty_end as usize],
            );
            Ok(())
        }

        fn record_readback(&mut self, readback_us: f64) {
            if self.recording_current_sample {
                self.layers[self.layer_count].stage_us[StageClass::Readback as usize] +=
                    readback_us;
            }
        }

        fn emit(&self, state: &DeviceState) {
            if self.executions == 0 {
                return;
            }
            let divisor = self.executions as f64;
            let layers = self
                .layers
                .iter()
                .enumerate()
                .map(|(layer, aggregate)| {
                    let stage_us = StageClass::ALL
                        .into_iter()
                        .map(|stage| {
                            (
                                stage.label().to_owned(),
                                serde_json::json!(aggregate.stage_us[stage as usize] / divisor),
                            )
                        })
                        .collect::<serde_json::Map<_, _>>();
                    let dispatches = self
                        .query
                        .dispatches
                        .iter()
                        .filter(|dispatch| dispatch.layer == layer)
                        .count();
                    serde_json::json!({
                        "layer": if layer == self.layer_count { serde_json::json!("final/readback") } else { serde_json::json!(layer) },
                        "stage_us_mean": stage_us,
                        "barrier_us_mean": aggregate.barrier_us / divisor,
                        "pipeline_barriers_per_execution": dispatches,
                        "descriptor_rebinds_per_execution": dispatches,
                    })
                })
                .collect::<Vec<_>>();
            let record = serde_json::json!({
                "schema": "synapse-vulkan-stage-profile-v1",
                "family": self.family,
                "shape": {"batch": self.batch, "seq": self.seq},
                "executions": self.executions,
                "warmup_executions_skipped": self.samples_seen - self.executions,
                "timestamp_period_ns": state.timestamp_period_ns,
                "timestamp_valid_bits": state.timestamp_valid_bits,
                "gpu_total_us_mean": self.gpu_total_us / divisor,
                "gpu_interval_residual_us_mean": self.gpu_interval_residual_us / divisor,
                "submit_wait_us_mean": self.submit_wait_us / divisor,
                "empty_timestamp_sandwich_us_mean": self.empty_sandwich_us / divisor,
                "dispatches_per_execution": self.query.dispatches.len(),
                "pipeline_barriers_per_execution": self.query.dispatches.len(),
                "descriptor_rebinds_per_execution": self.query.dispatches.len(),
                "layers": layers,
            });
            let line = match serde_json::to_string(&record) {
                Ok(line) => line,
                Err(error) => {
                    eprintln!("Vulkan profile serialization failed: {error}");
                    return;
                }
            };
            if let Some(path) = &state.profile_output {
                let result = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .and_then(|mut file| writeln!(file, "{line}"));
                if let Err(error) = result {
                    eprintln!(
                        "Vulkan profile write failed for {}: {error}",
                        path.display()
                    );
                }
            } else {
                eprintln!("VULKAN_PROFILE {line}");
            }
        }
    }

    struct Recorder<'a> {
        state: &'a DeviceState,
        command: vk::CommandBuffer,
        descriptor_pool: vk::DescriptorPool,
        descriptor_sets: Vec<vk::DescriptorSet>,
        pipelines: &'a Pipelines,
        profile: Option<QueryRecording>,
    }

    impl Recorder<'_> {
        fn dispatch<T: Copy>(
            &mut self,
            pipeline: vk::Pipeline,
            buffers: &[&Buffer],
            params: &T,
            groups: [u32; 3],
            layer: usize,
            stage: StageClass,
        ) -> Result<()> {
            let layouts = [self.state.descriptor_layout];
            let set = unsafe {
                self.state.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&layouts),
                )?[0]
            };
            self.descriptor_sets.push(set);
            let infos = buffers
                .iter()
                .map(|buffer| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(buffer.buffer)
                        .range(vk::WHOLE_SIZE)]
                })
                .collect::<Vec<_>>();
            let writes = infos
                .iter()
                .enumerate()
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect::<Vec<_>>();
            let queries = self
                .profile
                .as_mut()
                .map(|profile| profile.reserve_dispatch(layer, stage))
                .transpose()?;
            unsafe {
                self.state.device.update_descriptor_sets(&writes, &[]);
                self.state.device.cmd_bind_pipeline(
                    self.command,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline,
                );
                self.state.device.cmd_bind_descriptor_sets(
                    self.command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.state.pipeline_layout,
                    0,
                    &[set],
                    &[],
                );
                self.state.device.cmd_push_constants(
                    self.command,
                    self.state.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes_of(params),
                );
                if let Some((start, _, _)) = queries {
                    self.state.device.cmd_write_timestamp(
                        self.command,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        self.profile.as_ref().expect("profile exists").pool,
                        start,
                    );
                }
                self.state
                    .device
                    .cmd_dispatch(self.command, groups[0], groups[1], groups[2]);
                if let Some((_, dispatch_end, _)) = queries {
                    self.state.device.cmd_write_timestamp(
                        self.command,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        self.profile.as_ref().expect("profile exists").pool,
                        dispatch_end,
                    );
                }
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                self.state.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
                if let Some((_, _, barrier_end)) = queries {
                    self.state.device.cmd_write_timestamp(
                        self.command,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        self.profile.as_ref().expect("profile exists").pool,
                        barrier_end,
                    );
                }
            }
            Ok(())
        }

        fn finish_profile(&mut self) -> Result<()> {
            if let Some(profile) = &mut self.profile {
                profile.finish(self.state, self.command)?;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn gemm(
            &mut self,
            a: &Buffer,
            b: &Buffer,
            c: &Buffer,
            m: usize,
            n: usize,
            k: usize,
            batch_count: usize,
            transpose_b: bool,
            layer: usize,
            stage: StageClass,
        ) -> Result<()> {
            self.gemm_offset(
                a,
                b,
                c,
                m,
                n,
                k,
                batch_count,
                transpose_b,
                0,
                0,
                0,
                layer,
                stage,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn gemm_offset(
            &mut self,
            a: &Buffer,
            b: &Buffer,
            c: &Buffer,
            m: usize,
            n: usize,
            k: usize,
            batch_count: usize,
            transpose_b: bool,
            a_offset: usize,
            b_offset: usize,
            c_offset: usize,
            layer: usize,
            stage: StageClass,
        ) -> Result<()> {
            let groups = [
                n.div_ceil(16) as u32,
                m.div_ceil(16) as u32,
                batch_count as u32,
            ];
            let params = GemmParams {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                batch_count: batch_count as u32,
                transpose_b: transpose_b as u32,
                edge_only: 0,
                a_offset: a_offset as u32,
                b_offset: b_offset as u32,
                c_offset: c_offset as u32,
            };
            // Cooperative loads are certified for transposed weights and QK. PV keeps its
            // row-major B operand on the shared plain kernel after the bounded wave-2 probe.
            if matches!(self.state.gemm, VulkanGemm::Cooperative) && transpose_b && k % 16 == 0 {
                self.dispatch(
                    self.pipelines
                        .cooperative
                        .context("cooperative pipeline missing")?,
                    &[a, b, c],
                    &params,
                    groups,
                    layer,
                    stage,
                )?;
                if m % 16 != 0 || n % 16 != 0 {
                    self.dispatch(
                        self.pipelines.plain,
                        &[a, b, c],
                        &GemmParams {
                            edge_only: 1,
                            ..params
                        },
                        groups,
                        layer,
                        stage,
                    )?;
                }
                Ok(())
            } else {
                self.dispatch(
                    self.pipelines.plain,
                    &[a, b, c],
                    &params,
                    groups,
                    layer,
                    stage,
                )
            }
        }
    }

    struct ShapePlan {
        state: Arc<DeviceState>,
        command_pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
        command: vk::CommandBuffer,
        fence: vk::Fence,
        pipelines: Pipelines,
        _descriptor_sets: Vec<vk::DescriptorSet>,
        activations: Activations,
        profile: Option<ProfileCapture>,
        batch: usize,
        hidden: usize,
    }

    impl ShapePlan {
        #[allow(clippy::too_many_arguments)]
        fn new(
            state: Arc<DeviceState>,
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            epsilon: f32,
            layers: &[DeviceLayer],
        ) -> Result<Self> {
            let started = Instant::now();
            let pipelines = Pipelines::create(&state)?;
            let pipeline_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let activations =
                Activations::new(state.clone(), batch, seq, hidden, heads, intermediate)?;
            let command_pool = unsafe {
                state.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(state.queue_family),
                    None,
                )?
            };
            let command = unsafe {
                state.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0]
            };
            let dispatch_count = layers.len() as u32 * 24 + 1;
            let descriptor_pool = unsafe {
                state.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(dispatch_count)
                        .pool_sizes(&[vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(dispatch_count * DESCRIPTOR_BINDINGS)]),
                    None,
                )?
            };
            let fence = unsafe {
                state
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)?
            };
            unsafe {
                state.device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
                )?;
            }
            let mut recorder = Recorder {
                state: &state,
                command,
                descriptor_pool,
                descriptor_sets: Vec::new(),
                pipelines: &pipelines,
                profile: QueryRecording::new(&state, command, dispatch_count)?,
            };
            let rows = batch * seq;
            let head_dim = hidden / heads;
            let current = &activations.input;
            let next = &activations.x1;
            for (layer_index, layer) in layers.iter().enumerate() {
                recorder.gemm(
                    current,
                    &layer.query.weight,
                    &activations.q_raw,
                    rows,
                    hidden,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmQkv,
                )?;
                recorder.gemm(
                    current,
                    &layer.key.weight,
                    &activations.k_raw,
                    rows,
                    hidden,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmQkv,
                )?;
                recorder.gemm(
                    current,
                    &layer.value.weight,
                    &activations.v_raw,
                    rows,
                    hidden,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmQkv,
                )?;
                recorder.dispatch(
                    pipelines.qkv,
                    &[
                        &activations.q_raw,
                        &activations.k_raw,
                        &activations.v_raw,
                        &layer.query.bias,
                        &layer.key.bias,
                        &layer.value.bias,
                        &activations.q,
                        &activations.k,
                        &activations.v,
                    ],
                    &FourU32 {
                        a: batch as u32,
                        b: seq as u32,
                        c: heads as u32,
                        d: head_dim as u32,
                    },
                    [(batch * heads * seq * head_dim).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::LayoutTranspose,
                )?;
                recorder.gemm(
                    &activations.q,
                    &activations.k,
                    &activations.scores_f32,
                    seq,
                    seq,
                    head_dim,
                    batch * heads,
                    true,
                    layer_index,
                    StageClass::GemmAttentionScores,
                )?;
                recorder.dispatch(
                    pipelines.softmax,
                    &[
                        &activations.scores_f32,
                        &activations.mask,
                        &activations.scores_f16,
                    ],
                    &SoftmaxParams {
                        batch: batch as u32,
                        heads: heads as u32,
                        seq: seq as u32,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                    },
                    [(batch * heads * seq) as u32, 1, 1],
                    layer_index,
                    StageClass::SoftmaxMask,
                )?;
                recorder.gemm(
                    &activations.scores_f16,
                    &activations.v,
                    &activations.attention_f32,
                    seq,
                    head_dim,
                    seq,
                    batch * heads,
                    false,
                    layer_index,
                    StageClass::GemmPv,
                )?;
                recorder.dispatch(
                    pipelines.transpose,
                    &[&activations.attention_f32, &activations.context_f16],
                    &FourU32 {
                        a: batch as u32,
                        b: seq as u32,
                        c: heads as u32,
                        d: head_dim as u32,
                    },
                    [(batch * seq * hidden).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::LayoutTranspose,
                )?;
                recorder.gemm(
                    &activations.context_f16,
                    &layer.attention_output.weight,
                    &activations.projected_f32,
                    rows,
                    hidden,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmOut,
                )?;
                recorder.dispatch(
                    pipelines.residual_norm,
                    &[
                        &activations.projected_f32,
                        &layer.attention_output.bias,
                        current,
                        &layer.attention_ln_weight,
                        &layer.attention_ln_bias,
                        next,
                    ],
                    &NormParams {
                        rows: rows as u32,
                        hidden: hidden as u32,
                        epsilon,
                    },
                    [rows as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.gemm(
                    next,
                    &layer.intermediate.weight,
                    &activations.intermediate_f32,
                    rows,
                    intermediate,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmMlpUp,
                )?;
                recorder.dispatch(
                    pipelines.gelu,
                    &[
                        &activations.intermediate_f32,
                        &layer.intermediate.bias,
                        &activations.intermediate_f16,
                    ],
                    &FourU32 {
                        a: rows as u32,
                        b: intermediate as u32,
                        c: 0,
                        d: 0,
                    },
                    [(rows * intermediate).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.gemm(
                    &activations.intermediate_f16,
                    &layer.output.weight,
                    &activations.ffn_f32,
                    rows,
                    hidden,
                    intermediate,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmMlpDown,
                )?;
                recorder.dispatch(
                    pipelines.residual_norm,
                    &[
                        &activations.ffn_f32,
                        &layer.output.bias,
                        next,
                        &layer.output_ln_weight,
                        &layer.output_ln_bias,
                        current,
                    ],
                    &NormParams {
                        rows: rows as u32,
                        hidden: hidden as u32,
                        epsilon,
                    },
                    [rows as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
            }
            recorder.dispatch(
                pipelines.pool,
                &[current, &activations.mask, &activations.pooled],
                &FourU32 {
                    a: batch as u32,
                    b: seq as u32,
                    c: hidden as u32,
                    d: 0,
                },
                [batch as u32, 1, 1],
                layers.len(),
                StageClass::Pointwise,
            )?;
            recorder.finish_profile()?;
            unsafe { state.device.end_command_buffer(command)? };
            let profile = recorder
                .profile
                .take()
                .map(|query| ProfileCapture::new(query, "MiniLM", batch, seq, layers.len()));
            let descriptor_sets = std::mem::take(&mut recorder.descriptor_sets);
            let descriptor_set_count = descriptor_sets.len();
            drop(recorder);
            eprintln!(
                "Vulkan shape {batch}x{seq}: gemm={} pipeline_creation_ms={pipeline_ms:.3} descriptor_sets={descriptor_set_count} command_buffers=1 encoder_resident=true",
                state.gemm.as_str(),
            );
            Ok(Self {
                state,
                command_pool,
                descriptor_pool,
                command,
                fence,
                pipelines,
                _descriptor_sets: descriptor_sets,
                activations,
                profile,
                batch,
                hidden,
            })
        }

        fn run(&mut self, hidden_states: &[f32], mask: &[u8]) -> Result<Vec<Vec<f32>>> {
            self.activations
                .input
                .write(&encode_f16_bits(hidden_states))?;
            let mask_u32 = mask
                .iter()
                .map(|value| u32::from(*value))
                .collect::<Vec<_>>();
            self.activations.mask.write(&mask_u32)?;
            let submit_started = Instant::now();
            unsafe {
                self.state.device.reset_fences(&[self.fence])?;
                self.state.device.queue_submit(
                    self.state.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.command])],
                    self.fence,
                )?;
                self.state
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)?;
            }
            if let Some(profile) = &mut self.profile {
                profile.record_queries(
                    &self.state,
                    submit_started.elapsed().as_secs_f64() * 1_000_000.0,
                )?;
            }
            let readback_started = Instant::now();
            let output = self.activations.pooled.read_f32(self.batch * self.hidden)?;
            if let Some(profile) = &mut self.profile {
                profile.record_readback(readback_started.elapsed().as_secs_f64() * 1_000_000.0);
            }
            Ok(output
                .chunks_exact(self.hidden)
                .map(<[f32]>::to_vec)
                .collect())
        }
    }

    impl Drop for ShapePlan {
        fn drop(&mut self) {
            unsafe {
                let _ = self.state.device.device_wait_idle();
                if let Some(profile) = &self.profile {
                    profile.emit(&self.state);
                    self.state
                        .device
                        .destroy_query_pool(profile.query.pool, None);
                }
                for pipeline in self.pipelines.all() {
                    self.state.device.destroy_pipeline(pipeline, None);
                }
                self.state.device.destroy_fence(self.fence, None);
                self.state
                    .device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.state
                    .device
                    .destroy_command_pool(self.command_pool, None);
            }
        }
    }

    pub struct VulkanContext {
        state: Arc<DeviceState>,
        layers: Vec<DeviceLayer>,
        model_shape: Option<(usize, usize, usize)>,
        plans: HashMap<(usize, usize), ShapePlan>,
    }

    impl VulkanContext {
        pub fn new(gemm: VulkanGemm, pipeline_cache_path: Option<PathBuf>) -> Result<Self> {
            eprintln!("Vulkan phase: create context gemm={}", gemm.as_str());
            Ok(Self {
                state: DeviceState::new(gemm, pipeline_cache_path)?,
                layers: Vec::new(),
                model_shape: None,
                plans: HashMap::new(),
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encoder_forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            layer_norm_eps: f32,
            layers: &[EncoderLayer],
        ) -> Result<Vec<Vec<f32>>> {
            ensure!(
                batch > 0 && seq > 0 && hidden % heads == 0,
                "invalid Vulkan encoder dimensions"
            );
            ensure!(
                hidden_states.len() == batch * seq * hidden,
                "Vulkan hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == batch * seq,
                "Vulkan mask shape mismatch"
            );
            if self.layers.is_empty() {
                let started = Instant::now();
                self.layers = layers
                    .iter()
                    .map(|layer| DeviceLayer::upload(self.state.clone(), layer))
                    .collect::<Result<Vec<_>>>()?;
                self.model_shape = Some((hidden, intermediate, layers.len()));
                let (allocations, allocated_bytes) = self.state.immutable_allocation_summary();
                eprintln!(
                    "Vulkan persistent weights: family=MiniLM upload_ms={:.3} layers={} hidden={} intermediate={} storage=f16 norm_params=fp32 allocations={allocations} allocated_bytes={allocated_bytes} placement=DEVICE_LOCAL host_visible=false upload=staging-copy",
                    started.elapsed().as_secs_f64() * 1_000.0,
                    layers.len(),
                    hidden,
                    intermediate
                );
            }
            ensure!(
                self.model_shape == Some((hidden, intermediate, layers.len())),
                "Vulkan context model dimensions changed"
            );
            let key = (batch, seq);
            if !self.plans.contains_key(&key) {
                let plan = ShapePlan::new(
                    self.state.clone(),
                    batch,
                    seq,
                    hidden,
                    heads,
                    intermediate,
                    layer_norm_eps,
                    &self.layers,
                )?;
                self.plans.insert(key, plan);
            }
            self.plans
                .get_mut(&key)
                .context("Vulkan shape plan missing after insertion")?
                .run(hidden_states, attention_mask)
        }
    }

    struct DeviceModernLayer {
        qkv_weight: Buffer,
        attention_output_weight: Buffer,
        attention_norm_weight: Buffer,
        mlp_input_weight: Buffer,
        mlp_output_weight: Buffer,
        mlp_norm_weight: Buffer,
        sliding_attention: bool,
        has_attention_norm: bool,
    }

    impl DeviceModernLayer {
        fn upload(
            state: Arc<DeviceState>,
            layer: &ModernBertLayer<'_>,
            hidden: usize,
        ) -> Result<Self> {
            let identity_weight = vec![1.0f32; hidden];
            Ok(Self {
                qkv_weight: Buffer::from_f16(state.clone(), layer.qkv_weight)?,
                attention_output_weight: Buffer::from_f16(
                    state.clone(),
                    layer.attention_output_weight,
                )?,
                attention_norm_weight: Buffer::from_f32(
                    state.clone(),
                    layer.attention_norm_weight.unwrap_or(&identity_weight),
                )?,
                mlp_input_weight: Buffer::from_f16(state.clone(), layer.mlp_input_weight)?,
                mlp_output_weight: Buffer::from_f16(state.clone(), layer.mlp_output_weight)?,
                mlp_norm_weight: Buffer::from_f32(state, layer.mlp_norm_weight)?,
                sliding_attention: layer.sliding_attention,
                has_attention_norm: layer.attention_norm_weight.is_some(),
            })
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct FamilyNormParams {
        rows: u32,
        width: u32,
        epsilon: f32,
        identity: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct ModernSoftmaxParams {
        batch: u32,
        heads: u32,
        seq: u32,
        scale: f32,
        sliding: u32,
    }

    struct ModernActivations {
        input: Buffer,
        mask: Buffer,
        x1: Buffer,
        normed: Buffer,
        qkv: Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        scores: Buffer,
        probabilities: Buffer,
        attention: Buffer,
        context: Buffer,
        projected: Buffer,
        mlp_projected: Buffer,
        activated: Buffer,
        band: Buffer,
        global_cos: Buffer,
        global_sin: Buffer,
        local_cos: Buffer,
        local_sin: Buffer,
    }

    impl ModernActivations {
        #[allow(clippy::too_many_arguments)]
        fn new(
            state: Arc<DeviceState>,
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            half_window: usize,
            global_theta: f32,
            local_theta: f32,
        ) -> Result<Self> {
            let rows = batch * seq;
            let hidden_values = rows * hidden;
            let score_values = batch * heads * seq * seq;
            let f16 = |count| Buffer::new(state.clone(), count * size_of::<u16>());
            let f32_buffer = |count| Buffer::new(state.clone(), count * size_of::<f32>());
            let mut band = vec![0.0f32; seq * seq];
            for query in 0..seq {
                for key in 0..seq {
                    if query.abs_diff(key) > half_window {
                        band[query * seq + key] = -10_000.0;
                    }
                }
            }
            let rope = |theta: f32| {
                let head_dim = hidden / heads;
                let mut cosine = vec![0.0f32; seq * head_dim];
                let mut sine = vec![0.0f32; seq * head_dim];
                for position in 0..seq {
                    for index in 0..head_dim / 2 {
                        let frequency = theta.powf(-((2 * index) as f32) / head_dim as f32);
                        let (sin, cos) = (position as f32 * frequency).sin_cos();
                        for offset in [index, index + head_dim / 2] {
                            cosine[position * head_dim + offset] = cos;
                            sine[position * head_dim + offset] = sin;
                        }
                    }
                }
                (cosine, sine)
            };
            let (global_cos, global_sin) = rope(global_theta);
            let (local_cos, local_sin) = rope(local_theta);
            Ok(Self {
                input: f16(hidden_values)?,
                mask: Buffer::new(state.clone(), rows * size_of::<u32>())?,
                x1: f16(hidden_values)?,
                normed: f16(hidden_values)?,
                qkv: f32_buffer(hidden_values * 3)?,
                q: f16(hidden_values)?,
                k: f16(hidden_values)?,
                v: f16(hidden_values)?,
                scores: f32_buffer(score_values)?,
                probabilities: f16(score_values)?,
                attention: f32_buffer(hidden_values)?,
                context: f16(hidden_values)?,
                projected: f32_buffer(hidden_values)?,
                mlp_projected: f32_buffer(rows * intermediate * 2)?,
                activated: f16(rows * intermediate)?,
                band: Buffer::from_f32(state.clone(), &band)?,
                global_cos: Buffer::from_f32(state.clone(), &global_cos)?,
                global_sin: Buffer::from_f32(state.clone(), &global_sin)?,
                local_cos: Buffer::from_f32(state.clone(), &local_cos)?,
                local_sin: Buffer::from_f32(state, &local_sin)?,
            })
        }
    }

    struct ModernShapePlan {
        state: Arc<DeviceState>,
        command_pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
        command: vk::CommandBuffer,
        fence: vk::Fence,
        pipelines: Pipelines,
        _descriptor_sets: Vec<vk::DescriptorSet>,
        activations: ModernActivations,
        profile: Option<ProfileCapture>,
        values: usize,
    }

    impl ModernShapePlan {
        #[allow(clippy::too_many_arguments)]
        fn new(
            state: Arc<DeviceState>,
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            epsilon: f32,
            global_theta: f32,
            local_theta: f32,
            half_window: usize,
            layers: &[DeviceModernLayer],
            final_norm: &Buffer,
        ) -> Result<Self> {
            let pipelines = Pipelines::create(&state)?;
            let activations = ModernActivations::new(
                state.clone(),
                batch,
                seq,
                hidden,
                heads,
                intermediate,
                half_window,
                global_theta,
                local_theta,
            )?;
            let command_pool = unsafe {
                state.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(state.queue_family),
                    None,
                )?
            };
            let command = unsafe {
                state.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0]
            };
            let max_sets = layers.len() as u32 * 20 + 1;
            let descriptor_pool = unsafe {
                state.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(max_sets)
                        .pool_sizes(&[vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(max_sets * DESCRIPTOR_BINDINGS)]),
                    None,
                )?
            };
            let fence = unsafe {
                state
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)?
            };
            unsafe {
                state
                    .device
                    .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
            }
            let mut recorder = Recorder {
                state: &state,
                command,
                descriptor_pool,
                descriptor_sets: Vec::new(),
                pipelines: &pipelines,
                profile: QueryRecording::new(&state, command, max_sets)?,
            };
            let rows = batch * seq;
            let head_dim = hidden / heads;
            for (layer_index, layer) in layers.iter().enumerate() {
                recorder.dispatch(
                    pipelines.layer_norm,
                    &[
                        &activations.input,
                        &layer.attention_norm_weight,
                        &activations.normed,
                    ],
                    &FamilyNormParams {
                        rows: rows as u32,
                        width: hidden as u32,
                        epsilon,
                        identity: u32::from(!layer.has_attention_norm),
                    },
                    [rows as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.gemm(
                    &activations.normed,
                    &layer.qkv_weight,
                    &activations.qkv,
                    rows,
                    hidden * 3,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmQkv,
                )?;
                let (cosine, sine) = if layer.sliding_attention {
                    (&activations.local_cos, &activations.local_sin)
                } else {
                    (&activations.global_cos, &activations.global_sin)
                };
                recorder.dispatch(
                    pipelines.modern_qkv_rope,
                    &[
                        &activations.qkv,
                        cosine,
                        sine,
                        &activations.q,
                        &activations.k,
                        &activations.v,
                    ],
                    &FourU32 {
                        a: batch as u32,
                        b: seq as u32,
                        c: heads as u32,
                        d: head_dim as u32,
                    },
                    [(batch * heads * seq * head_dim).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::LayoutTranspose,
                )?;
                recorder.gemm(
                    &activations.q,
                    &activations.k,
                    &activations.scores,
                    seq,
                    seq,
                    head_dim,
                    batch * heads,
                    true,
                    layer_index,
                    StageClass::GemmAttentionScores,
                )?;
                recorder.dispatch(
                    pipelines.modern_softmax,
                    &[
                        &activations.scores,
                        &activations.mask,
                        &activations.band,
                        &activations.probabilities,
                    ],
                    &ModernSoftmaxParams {
                        batch: batch as u32,
                        heads: heads as u32,
                        seq: seq as u32,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                        sliding: u32::from(layer.sliding_attention),
                    },
                    [(batch * heads * seq) as u32, 1, 1],
                    layer_index,
                    StageClass::SoftmaxMask,
                )?;
                recorder.gemm(
                    &activations.probabilities,
                    &activations.v,
                    &activations.attention,
                    seq,
                    head_dim,
                    seq,
                    batch * heads,
                    false,
                    layer_index,
                    StageClass::GemmPv,
                )?;
                recorder.dispatch(
                    pipelines.transpose,
                    &[&activations.attention, &activations.context],
                    &FourU32 {
                        a: batch as u32,
                        b: seq as u32,
                        c: heads as u32,
                        d: head_dim as u32,
                    },
                    [(rows * hidden).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::LayoutTranspose,
                )?;
                recorder.gemm(
                    &activations.context,
                    &layer.attention_output_weight,
                    &activations.projected,
                    rows,
                    hidden,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmOut,
                )?;
                recorder.dispatch(
                    pipelines.add_residual,
                    &[&activations.projected, &activations.input, &activations.x1],
                    &FourU32 {
                        a: (rows * hidden) as u32,
                        b: 0,
                        c: 0,
                        d: 0,
                    },
                    [(rows * hidden).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.dispatch(
                    pipelines.layer_norm,
                    &[&activations.x1, &layer.mlp_norm_weight, &activations.normed],
                    &FamilyNormParams {
                        rows: rows as u32,
                        width: hidden as u32,
                        epsilon,
                        identity: 0,
                    },
                    [rows as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.gemm(
                    &activations.normed,
                    &layer.mlp_input_weight,
                    &activations.mlp_projected,
                    rows,
                    intermediate * 2,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmMlpUp,
                )?;
                recorder.dispatch(
                    pipelines.geglu,
                    &[&activations.mlp_projected, &activations.activated],
                    &FourU32 {
                        a: rows as u32,
                        b: intermediate as u32,
                        c: 0,
                        d: 0,
                    },
                    [(rows * intermediate).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.gemm(
                    &activations.activated,
                    &layer.mlp_output_weight,
                    &activations.projected,
                    rows,
                    hidden,
                    intermediate,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmMlpDown,
                )?;
                recorder.dispatch(
                    pipelines.add_residual,
                    &[&activations.projected, &activations.x1, &activations.input],
                    &FourU32 {
                        a: (rows * hidden) as u32,
                        b: 0,
                        c: 0,
                        d: 0,
                    },
                    [(rows * hidden).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
            }
            recorder.dispatch(
                pipelines.layer_norm,
                &[&activations.input, final_norm, &activations.x1],
                &FamilyNormParams {
                    rows: rows as u32,
                    width: hidden as u32,
                    epsilon,
                    identity: 0,
                },
                [rows as u32, 1, 1],
                layers.len(),
                StageClass::Pointwise,
            )?;
            recorder.finish_profile()?;
            unsafe {
                state.device.end_command_buffer(command)?;
            }
            let profile = recorder.profile.take().map(|query| {
                ProfileCapture::new(query, "gte-modernbert", batch, seq, layers.len())
            });
            let descriptor_sets = std::mem::take(&mut recorder.descriptor_sets);
            drop(recorder);
            eprintln!("Vulkan ModernBERT shape {batch}x{seq}: gemm={} encoder_resident=true dual_theta=true local_mask=content", state.gemm.as_str());
            Ok(Self {
                state,
                command_pool,
                descriptor_pool,
                command,
                fence,
                pipelines,
                _descriptor_sets: descriptor_sets,
                activations,
                profile,
                values: rows * hidden,
            })
        }

        fn run(&mut self, hidden_states: &mut [f32], mask: &[u8]) -> Result<()> {
            self.activations
                .input
                .write(&encode_f16_bits(hidden_states))?;
            self.activations.mask.write(
                &mask
                    .iter()
                    .map(|value| u32::from(*value))
                    .collect::<Vec<_>>(),
            )?;
            let submit_started = Instant::now();
            unsafe {
                self.state.device.reset_fences(&[self.fence])?;
                self.state.device.queue_submit(
                    self.state.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.command])],
                    self.fence,
                )?;
                self.state
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)?;
            }
            if let Some(profile) = &mut self.profile {
                profile.record_queries(
                    &self.state,
                    submit_started.elapsed().as_secs_f64() * 1_000_000.0,
                )?;
            }
            let readback_started = Instant::now();
            let output = self.activations.x1.read_u16(self.values)?;
            if let Some(profile) = &mut self.profile {
                profile.record_readback(readback_started.elapsed().as_secs_f64() * 1_000_000.0);
            }
            hidden_states.copy_from_slice(&decode_f16_bits(&output));
            Ok(())
        }
    }

    impl Drop for ModernShapePlan {
        fn drop(&mut self) {
            unsafe {
                let _ = self.state.device.device_wait_idle();
                if let Some(profile) = &self.profile {
                    profile.emit(&self.state);
                    self.state
                        .device
                        .destroy_query_pool(profile.query.pool, None);
                }
                for pipeline in self.pipelines.all() {
                    self.state.device.destroy_pipeline(pipeline, None);
                }
                self.state.device.destroy_fence(self.fence, None);
                self.state
                    .device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.state
                    .device
                    .destroy_command_pool(self.command_pool, None);
            }
        }
    }

    pub struct ModernBertContext {
        state: Arc<DeviceState>,
        layers: Vec<DeviceModernLayer>,
        final_norm: Option<Buffer>,
        model_shape: Option<(usize, usize, usize)>,
        plans: HashMap<(usize, usize), ModernShapePlan>,
    }

    impl ModernBertContext {
        pub fn new(gemm: VulkanGemm, pipeline_cache: Option<PathBuf>) -> Result<Self> {
            Ok(Self {
                state: DeviceState::new(gemm, pipeline_cache)?,
                layers: Vec::new(),
                final_norm: None,
                model_shape: None,
                plans: HashMap::new(),
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            epsilon: f32,
            global_theta: f32,
            local_theta: f32,
            half_window: usize,
            layers: &[ModernBertLayer<'_>],
            final_norm: &[f32],
        ) -> Result<()> {
            ensure!(
                hidden_states.len() == batch * seq * hidden,
                "ModernBERT Vulkan hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == batch * seq,
                "ModernBERT Vulkan mask shape mismatch"
            );
            if self.layers.is_empty() {
                let started = Instant::now();
                self.layers = layers
                    .iter()
                    .map(|layer| DeviceModernLayer::upload(self.state.clone(), layer, hidden))
                    .collect::<Result<_>>()?;
                self.final_norm = Some(Buffer::from_f32(self.state.clone(), final_norm)?);
                self.model_shape = Some((hidden, intermediate, layers.len()));
                let (allocations, allocated_bytes) = self.state.immutable_allocation_summary();
                eprintln!(
                    "Vulkan persistent weights: family=gte-modernbert upload_ms={:.3} layers={} hidden={} intermediate={} storage=f16 norm_params=fp32 allocations={allocations} allocated_bytes={allocated_bytes} placement=DEVICE_LOCAL host_visible=false upload=staging-copy",
                    started.elapsed().as_secs_f64() * 1_000.0,
                    layers.len(),
                    hidden,
                    intermediate
                );
            }
            ensure!(
                self.model_shape == Some((hidden, intermediate, layers.len())),
                "ModernBERT Vulkan model dimensions changed"
            );
            let key = (batch, seq);
            if !self.plans.contains_key(&key) {
                let plan = ModernShapePlan::new(
                    self.state.clone(),
                    batch,
                    seq,
                    hidden,
                    heads,
                    intermediate,
                    epsilon,
                    global_theta,
                    local_theta,
                    half_window,
                    &self.layers,
                    self.final_norm
                        .as_ref()
                        .context("ModernBERT Vulkan final norm missing")?,
                )?;
                self.plans.insert(key, plan);
            }
            self.plans
                .get_mut(&key)
                .context("ModernBERT Vulkan shape plan missing")?
                .run(hidden_states, attention_mask)
        }
    }

    struct DeviceQwenLayer {
        input_norm: Buffer,
        post_attention_norm: Buffer,
        q_weight: Buffer,
        q_norm: Buffer,
        k_weight: Buffer,
        k_norm: Buffer,
        v_weight: Buffer,
        o_weight: Buffer,
        gate_weight: Buffer,
        up_weight: Buffer,
        down_weight: Buffer,
    }

    impl DeviceQwenLayer {
        fn upload(state: Arc<DeviceState>, layer: &Qwen3Layer<'_>) -> Result<Self> {
            Ok(Self {
                input_norm: Buffer::from_f32(state.clone(), layer.input_norm)?,
                post_attention_norm: Buffer::from_f32(state.clone(), layer.post_attention_norm)?,
                q_weight: Buffer::from_f16(state.clone(), layer.q_weight)?,
                q_norm: Buffer::from_f32(state.clone(), layer.q_norm)?,
                k_weight: Buffer::from_f16(state.clone(), layer.k_weight)?,
                k_norm: Buffer::from_f32(state.clone(), layer.k_norm)?,
                v_weight: Buffer::from_f16(state.clone(), layer.v_weight)?,
                o_weight: Buffer::from_f16(state.clone(), layer.o_weight)?,
                gate_weight: Buffer::from_f16(state.clone(), layer.gate_weight)?,
                up_weight: Buffer::from_f16(state.clone(), layer.up_weight)?,
                down_weight: Buffer::from_f16(state, layer.down_weight)?,
            })
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct QwenHeadParams {
        batch: u32,
        seq: u32,
        heads: u32,
        head_dim: u32,
        epsilon: f32,
        groups: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct QwenContextParams {
        batch: u32,
        seq: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    }

    struct QwenActivations {
        input: Buffer,
        mask: Buffer,
        x1: Buffer,
        normed: Buffer,
        q_raw: Buffer,
        k_raw: Buffer,
        v_raw: Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        scores: Buffer,
        probabilities: Buffer,
        attention: Buffer,
        context: Buffer,
        projected: Buffer,
        gate: Buffer,
        up: Buffer,
        activated: Buffer,
        cosine: Buffer,
        sine: Buffer,
    }

    impl QwenActivations {
        #[allow(clippy::too_many_arguments)]
        fn new(
            state: Arc<DeviceState>,
            batch: usize,
            seq: usize,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            theta: f32,
        ) -> Result<Self> {
            let rows = batch * seq;
            let hidden_values = rows * hidden;
            let query_values = rows * query_heads * head_dim;
            let kv_values = rows * kv_heads * head_dim;
            let score_values = batch * query_heads * seq * seq;
            let f16 = |count| Buffer::new(state.clone(), count * size_of::<u16>());
            let f32_buffer = |count| Buffer::new(state.clone(), count * size_of::<f32>());
            let mut cosine = vec![0.0f32; seq * head_dim];
            let mut sine = vec![0.0f32; seq * head_dim];
            for position in 0..seq {
                for index in 0..head_dim / 2 {
                    let frequency = theta.powf(-((2 * index) as f32) / head_dim as f32);
                    let (sin, cos) = (position as f32 * frequency).sin_cos();
                    for offset in [index, index + head_dim / 2] {
                        cosine[position * head_dim + offset] = cos;
                        sine[position * head_dim + offset] = sin;
                    }
                }
            }
            Ok(Self {
                input: f16(hidden_values)?,
                mask: Buffer::new(state.clone(), rows * size_of::<u32>())?,
                x1: f16(hidden_values)?,
                normed: f16(hidden_values)?,
                q_raw: f32_buffer(query_values)?,
                k_raw: f32_buffer(kv_values)?,
                v_raw: f32_buffer(kv_values)?,
                q: f16(query_values)?,
                k: f16(kv_values)?,
                v: f16(kv_values)?,
                scores: f32_buffer(score_values)?,
                probabilities: f16(score_values)?,
                attention: f32_buffer(query_values)?,
                context: f16(query_values)?,
                projected: f32_buffer(hidden_values)?,
                gate: f32_buffer(rows * intermediate)?,
                up: f32_buffer(rows * intermediate)?,
                activated: f16(rows * intermediate)?,
                cosine: Buffer::from_f32(state.clone(), &cosine)?,
                sine: Buffer::from_f32(state, &sine)?,
            })
        }
    }

    struct QwenShapePlan {
        state: Arc<DeviceState>,
        command_pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
        command: vk::CommandBuffer,
        fence: vk::Fence,
        pipelines: Pipelines,
        _descriptor_sets: Vec<vk::DescriptorSet>,
        activations: QwenActivations,
        profile: Option<ProfileCapture>,
        values: usize,
    }

    impl QwenShapePlan {
        #[allow(clippy::too_many_arguments)]
        fn new(
            state: Arc<DeviceState>,
            batch: usize,
            seq: usize,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            epsilon: f32,
            theta: f32,
            layers: &[DeviceQwenLayer],
            final_norm: &Buffer,
        ) -> Result<Self> {
            let pipelines = Pipelines::create(&state)?;
            let activations = QwenActivations::new(
                state.clone(),
                batch,
                seq,
                hidden,
                query_heads,
                kv_heads,
                head_dim,
                intermediate,
                theta,
            )?;
            let command_pool = unsafe {
                state.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(state.queue_family),
                    None,
                )?
            };
            let command = unsafe {
                state.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0]
            };
            let max_sets = layers.len() as u32 * 24 + 1;
            let descriptor_pool = unsafe {
                state.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(max_sets)
                        .pool_sizes(&[vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(max_sets * DESCRIPTOR_BINDINGS)]),
                    None,
                )?
            };
            let fence = unsafe {
                state
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)?
            };
            unsafe {
                state
                    .device
                    .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
            }
            let mut recorder = Recorder {
                state: &state,
                command,
                descriptor_pool,
                descriptor_sets: Vec::new(),
                pipelines: &pipelines,
                profile: QueryRecording::new(&state, command, max_sets)?,
            };
            let rows = batch * seq;
            let query_width = query_heads * head_dim;
            let kv_width = kv_heads * head_dim;
            let groups = query_heads / kv_heads;
            let group_batches = batch * kv_heads;
            let query_group_values = batch * kv_heads * seq * head_dim;
            let score_group_values = batch * kv_heads * seq * seq;
            for (layer_index, layer) in layers.iter().enumerate() {
                recorder.dispatch(
                    pipelines.rms_norm,
                    &[&activations.input, &layer.input_norm, &activations.normed],
                    &FamilyNormParams {
                        rows: rows as u32,
                        width: hidden as u32,
                        epsilon,
                        identity: 0,
                    },
                    [rows as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.gemm(
                    &activations.normed,
                    &layer.q_weight,
                    &activations.q_raw,
                    rows,
                    query_width,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmQkv,
                )?;
                recorder.gemm(
                    &activations.normed,
                    &layer.k_weight,
                    &activations.k_raw,
                    rows,
                    kv_width,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmQkv,
                )?;
                recorder.gemm(
                    &activations.normed,
                    &layer.v_weight,
                    &activations.v_raw,
                    rows,
                    kv_width,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmQkv,
                )?;
                recorder.dispatch(
                    pipelines.qwen_head_norm_rope,
                    &[
                        &activations.q_raw,
                        &layer.q_norm,
                        &activations.cosine,
                        &activations.sine,
                        &activations.q,
                    ],
                    &QwenHeadParams {
                        batch: batch as u32,
                        seq: seq as u32,
                        heads: query_heads as u32,
                        head_dim: head_dim as u32,
                        epsilon,
                        groups: groups as u32,
                    },
                    [(rows * query_heads) as u32, 1, 1],
                    layer_index,
                    StageClass::LayoutTranspose,
                )?;
                recorder.dispatch(
                    pipelines.qwen_head_norm_rope,
                    &[
                        &activations.k_raw,
                        &layer.k_norm,
                        &activations.cosine,
                        &activations.sine,
                        &activations.k,
                    ],
                    &QwenHeadParams {
                        batch: batch as u32,
                        seq: seq as u32,
                        heads: kv_heads as u32,
                        head_dim: head_dim as u32,
                        epsilon,
                        groups: 1,
                    },
                    [(rows * kv_heads) as u32, 1, 1],
                    layer_index,
                    StageClass::LayoutTranspose,
                )?;
                recorder.dispatch(
                    pipelines.qwen_value_transpose,
                    &[&activations.v_raw, &activations.v],
                    &FourU32 {
                        a: batch as u32,
                        b: seq as u32,
                        c: kv_heads as u32,
                        d: head_dim as u32,
                    },
                    [(rows * kv_width).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::LayoutTranspose,
                )?;
                for group in 0..groups {
                    recorder.gemm_offset(
                        &activations.q,
                        &activations.k,
                        &activations.scores,
                        seq,
                        seq,
                        head_dim,
                        group_batches,
                        true,
                        group * query_group_values,
                        0,
                        group * score_group_values,
                        layer_index,
                        StageClass::GemmAttentionScores,
                    )?;
                }
                recorder.dispatch(
                    pipelines.qwen_causal_softmax,
                    &[
                        &activations.scores,
                        &activations.mask,
                        &activations.probabilities,
                    ],
                    &SoftmaxParams {
                        batch: batch as u32,
                        heads: kv_heads as u32,
                        seq: seq as u32,
                        scale: 1.0 / (head_dim as f32).sqrt(),
                    },
                    [(batch * query_heads * seq) as u32, 1, 1],
                    layer_index,
                    StageClass::SoftmaxMask,
                )?;
                for group in 0..groups {
                    recorder.gemm_offset(
                        &activations.probabilities,
                        &activations.v,
                        &activations.attention,
                        seq,
                        head_dim,
                        seq,
                        group_batches,
                        false,
                        group * score_group_values,
                        0,
                        group * query_group_values,
                        layer_index,
                        StageClass::GemmPv,
                    )?;
                }
                // Query width is 2048 for this checkpoint, twice hidden. Dispatching from hidden
                // would silently leave half of every multi-row batch untouched.
                recorder.dispatch(
                    pipelines.qwen_context_transpose,
                    &[&activations.attention, &activations.context],
                    &QwenContextParams {
                        batch: batch as u32,
                        seq: seq as u32,
                        query_heads: query_heads as u32,
                        kv_heads: kv_heads as u32,
                        head_dim: head_dim as u32,
                    },
                    [(rows * query_width).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::LayoutTranspose,
                )?;
                recorder.gemm(
                    &activations.context,
                    &layer.o_weight,
                    &activations.projected,
                    rows,
                    hidden,
                    query_width,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmOut,
                )?;
                recorder.dispatch(
                    pipelines.add_residual,
                    &[&activations.projected, &activations.input, &activations.x1],
                    &FourU32 {
                        a: (rows * hidden) as u32,
                        b: 0,
                        c: 0,
                        d: 0,
                    },
                    [(rows * hidden).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.dispatch(
                    pipelines.rms_norm,
                    &[
                        &activations.x1,
                        &layer.post_attention_norm,
                        &activations.normed,
                    ],
                    &FamilyNormParams {
                        rows: rows as u32,
                        width: hidden as u32,
                        epsilon,
                        identity: 0,
                    },
                    [rows as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.gemm(
                    &activations.normed,
                    &layer.gate_weight,
                    &activations.gate,
                    rows,
                    intermediate,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmMlpUp,
                )?;
                recorder.gemm(
                    &activations.normed,
                    &layer.up_weight,
                    &activations.up,
                    rows,
                    intermediate,
                    hidden,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmMlpUp,
                )?;
                recorder.dispatch(
                    pipelines.swiglu,
                    &[&activations.gate, &activations.up, &activations.activated],
                    &FourU32 {
                        a: (rows * intermediate) as u32,
                        b: 0,
                        c: 0,
                        d: 0,
                    },
                    [(rows * intermediate).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
                recorder.gemm(
                    &activations.activated,
                    &layer.down_weight,
                    &activations.projected,
                    rows,
                    hidden,
                    intermediate,
                    1,
                    true,
                    layer_index,
                    StageClass::GemmMlpDown,
                )?;
                recorder.dispatch(
                    pipelines.add_residual,
                    &[&activations.projected, &activations.x1, &activations.input],
                    &FourU32 {
                        a: (rows * hidden) as u32,
                        b: 0,
                        c: 0,
                        d: 0,
                    },
                    [(rows * hidden).div_ceil(256) as u32, 1, 1],
                    layer_index,
                    StageClass::Pointwise,
                )?;
            }
            recorder.dispatch(
                pipelines.rms_norm,
                &[&activations.input, final_norm, &activations.x1],
                &FamilyNormParams {
                    rows: rows as u32,
                    width: hidden as u32,
                    epsilon,
                    identity: 0,
                },
                [rows as u32, 1, 1],
                layers.len(),
                StageClass::Pointwise,
            )?;
            recorder.finish_profile()?;
            unsafe {
                state.device.end_command_buffer(command)?;
            }
            let profile = recorder.profile.take().map(|query| {
                ProfileCapture::new(query, "Qwen3-Embedding-0.6B", batch, seq, layers.len())
            });
            let descriptor_sets = std::mem::take(&mut recorder.descriptor_sets);
            drop(recorder);
            eprintln!("Vulkan Qwen3 shape {batch}x{seq}: gemm={} encoder_resident=true gqa=two-group-strided kv_repeat_bytes=0 query_width={query_width}", state.gemm.as_str());
            Ok(Self {
                state,
                command_pool,
                descriptor_pool,
                command,
                fence,
                pipelines,
                _descriptor_sets: descriptor_sets,
                activations,
                profile,
                values: rows * hidden,
            })
        }

        fn run(&mut self, hidden_states: &mut [f32], mask: &[u8]) -> Result<()> {
            self.activations
                .input
                .write(&encode_f16_bits(hidden_states))?;
            self.activations.mask.write(
                &mask
                    .iter()
                    .map(|value| u32::from(*value))
                    .collect::<Vec<_>>(),
            )?;
            let submit_started = Instant::now();
            unsafe {
                self.state.device.reset_fences(&[self.fence])?;
                self.state.device.queue_submit(
                    self.state.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.command])],
                    self.fence,
                )?;
                self.state
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)?;
            }
            if let Some(profile) = &mut self.profile {
                profile.record_queries(
                    &self.state,
                    submit_started.elapsed().as_secs_f64() * 1_000_000.0,
                )?;
            }
            let readback_started = Instant::now();
            let output = self.activations.x1.read_u16(self.values)?;
            if let Some(profile) = &mut self.profile {
                profile.record_readback(readback_started.elapsed().as_secs_f64() * 1_000_000.0);
            }
            hidden_states.copy_from_slice(&decode_f16_bits(&output));
            Ok(())
        }
    }

    impl Drop for QwenShapePlan {
        fn drop(&mut self) {
            unsafe {
                let _ = self.state.device.device_wait_idle();
                if let Some(profile) = &self.profile {
                    profile.emit(&self.state);
                    self.state
                        .device
                        .destroy_query_pool(profile.query.pool, None);
                }
                for pipeline in self.pipelines.all() {
                    self.state.device.destroy_pipeline(pipeline, None);
                }
                self.state.device.destroy_fence(self.fence, None);
                self.state
                    .device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.state
                    .device
                    .destroy_command_pool(self.command_pool, None);
            }
        }
    }

    pub struct Qwen3Context {
        state: Arc<DeviceState>,
        layers: Vec<DeviceQwenLayer>,
        final_norm: Option<Buffer>,
        model_shape: Option<(usize, usize, usize, usize, usize)>,
        plans: HashMap<(usize, usize), QwenShapePlan>,
    }

    impl Qwen3Context {
        pub fn new(gemm: VulkanGemm, pipeline_cache: Option<PathBuf>) -> Result<Self> {
            Ok(Self {
                state: DeviceState::new(gemm, pipeline_cache)?,
                layers: Vec::new(),
                final_norm: None,
                model_shape: None,
                plans: HashMap::new(),
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            epsilon: f32,
            theta: f32,
            layers: &[Qwen3Layer<'_>],
            final_norm: &[f32],
        ) -> Result<()> {
            ensure!(
                hidden_states.len() == batch * seq * hidden,
                "Qwen3 Vulkan hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == batch * seq,
                "Qwen3 Vulkan mask shape mismatch"
            );
            ensure!(
                query_heads % kv_heads == 0,
                "Qwen3 Vulkan invalid GQA heads"
            );
            if self.layers.is_empty() {
                let started = Instant::now();
                self.layers = layers
                    .iter()
                    .map(|layer| DeviceQwenLayer::upload(self.state.clone(), layer))
                    .collect::<Result<_>>()?;
                self.final_norm = Some(Buffer::from_f32(self.state.clone(), final_norm)?);
                self.model_shape =
                    Some((hidden, query_heads, kv_heads, intermediate, layers.len()));
                let (allocations, allocated_bytes) = self.state.immutable_allocation_summary();
                eprintln!(
                    "Vulkan persistent weights: family=Qwen3-Embedding-0.6B upload_ms={:.3} layers={} hidden={} intermediate={} storage=f16 norm_params=fp32 allocations={allocations} allocated_bytes={allocated_bytes} placement=DEVICE_LOCAL host_visible=false upload=staging-copy",
                    started.elapsed().as_secs_f64() * 1_000.0,
                    layers.len(),
                    hidden,
                    intermediate
                );
            }
            ensure!(
                self.model_shape
                    == Some((hidden, query_heads, kv_heads, intermediate, layers.len())),
                "Qwen3 Vulkan model dimensions changed"
            );
            let key = (batch, seq);
            if !self.plans.contains_key(&key) {
                let plan = QwenShapePlan::new(
                    self.state.clone(),
                    batch,
                    seq,
                    hidden,
                    query_heads,
                    kv_heads,
                    head_dim,
                    intermediate,
                    epsilon,
                    theta,
                    &self.layers,
                    self.final_norm
                        .as_ref()
                        .context("Qwen3 Vulkan final norm missing")?,
                )?;
                self.plans.insert(key, plan);
            }
            self.plans
                .get_mut(&key)
                .context("Qwen3 Vulkan shape plan missing")?
                .run(hidden_states, attention_mask)
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct DecodeHeadParams {
        heads: u32,
        head_dim: u32,
        position: u32,
        cache_stride: u32,
        epsilon: f32,
        write_cache: u32,
        unused0: u32,
        unused1: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct DecodeAttentionParams {
        position: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        unused0: u32,
        unused1: u32,
        unused2: u32,
    }

    struct DeviceDecodeWeight {
        f16: Option<Buffer>,
        q8_0: Option<Buffer>,
    }

    impl DeviceDecodeWeight {
        fn upload(state: Arc<DeviceState>, weight: &Weight) -> Result<Self> {
            Self::upload_parts(
                state,
                &weight.tensor.data,
                weight.q8_0.as_ref().map(|quantized| quantized.as_bytes()),
            )
        }

        fn upload_parts(
            state: Arc<DeviceState>,
            values: &[f32],
            q8_0: Option<&[u8]>,
        ) -> Result<Self> {
            match q8_0 {
                Some(bytes) => Ok(Self {
                    f16: None,
                    q8_0: Some(Buffer::from_slice(state, bytes)?),
                }),
                None => Ok(Self {
                    f16: Some(Buffer::from_f16(state, values)?),
                    q8_0: None,
                }),
            }
        }
    }

    struct DeviceQwenDecodeLayer {
        input_norm: Buffer,
        post_attention_norm: Buffer,
        q_weight: DeviceDecodeWeight,
        q_norm: Buffer,
        k_weight: DeviceDecodeWeight,
        k_norm: Buffer,
        v_weight: DeviceDecodeWeight,
        o_weight: DeviceDecodeWeight,
        gate_weight: DeviceDecodeWeight,
        up_weight: DeviceDecodeWeight,
        down_weight: DeviceDecodeWeight,
        key_cache: Buffer,
        value_cache: Buffer,
    }

    struct QwenDecodeActivations {
        x0: Buffer,
        x1: Buffer,
        normed: Buffer,
        q_raw: Buffer,
        k_raw: Buffer,
        v_raw: Buffer,
        q: Buffer,
        attention: Buffer,
        projected: Buffer,
        gate: Buffer,
        up: Buffer,
        activated: Buffer,
        cosine: Buffer,
        sine: Buffer,
        scores: Buffer,
        logits: Buffer,
    }

    impl QwenDecodeActivations {
        #[allow(clippy::too_many_arguments)]
        fn new(
            state: Arc<DeviceState>,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            vocab: usize,
            capacity: usize,
        ) -> Result<Self> {
            let query_width = query_heads * head_dim;
            let kv_width = kv_heads * head_dim;
            let f16 = |count| Buffer::new(state.clone(), count * size_of::<u16>());
            let f32_buffer = |count| Buffer::new(state.clone(), count * size_of::<f32>());
            Ok(Self {
                x0: f16(hidden)?,
                x1: f16(hidden)?,
                normed: f16(hidden)?,
                q_raw: f32_buffer(query_width)?,
                k_raw: f32_buffer(kv_width)?,
                v_raw: f32_buffer(kv_width)?,
                q: f16(query_width)?,
                attention: f16(query_width)?,
                projected: f32_buffer(hidden)?,
                gate: f32_buffer(intermediate)?,
                up: f32_buffer(intermediate)?,
                activated: f16(intermediate)?,
                cosine: f32_buffer(head_dim)?,
                sine: f32_buffer(head_dim)?,
                scores: f32_buffer(query_heads * capacity)?,
                logits: f32_buffer(vocab)?,
            })
        }
    }

    /// Wave-5 batched (mat-mat) activation buffers, sized for the maximum
    /// column count K=16. Each buffer holds K columns laid out as
    /// `[K][width]`. The per-column layout matches the single-token buffers so
    /// column k's slice is bit-identical to a standalone single-token step.
    /// These are allocated lazily on first `verify_batch` and never on the
    /// default single-token path, so the existing decode path is unperturbed.
    struct QwenDecodeBatchActivations {
        x0: Buffer,
        x1: Buffer,
        normed: Buffer,
        q_raw: Buffer,
        k_raw: Buffer,
        v_raw: Buffer,
        q: Buffer,
        attention: Buffer,
        projected: Buffer,
        gate: Buffer,
        up: Buffer,
        activated: Buffer,
        cosine: Buffer,
        sine: Buffer,
        scores: Buffer,
        logits: Buffer,
    }

    impl QwenDecodeBatchActivations {
        #[allow(clippy::too_many_arguments)]
        fn new(
            state: Arc<DeviceState>,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            vocab: usize,
            capacity: usize,
            max_k: usize,
        ) -> Result<Self> {
            let query_width = query_heads * head_dim;
            let kv_width = kv_heads * head_dim;
            let f16 = |count| Buffer::new(state.clone(), count * size_of::<u16>());
            let f32_buffer = |count| Buffer::new(state.clone(), count * size_of::<f32>());
            Ok(Self {
                x0: f16(max_k * hidden)?,
                x1: f16(max_k * hidden)?,
                normed: f16(max_k * hidden)?,
                q_raw: f32_buffer(max_k * query_width)?,
                k_raw: f32_buffer(max_k * kv_width)?,
                v_raw: f32_buffer(max_k * kv_width)?,
                q: f16(max_k * query_width)?,
                attention: f16(max_k * query_width)?,
                projected: f32_buffer(max_k * hidden)?,
                gate: f32_buffer(max_k * intermediate)?,
                up: f32_buffer(max_k * intermediate)?,
                activated: f16(max_k * intermediate)?,
                cosine: f32_buffer(max_k * head_dim)?,
                sine: f32_buffer(max_k * head_dim)?,
                scores: f32_buffer(max_k * query_heads * capacity)?,
                logits: f32_buffer(max_k * vocab)?,
            })
        }
    }

    /// Plain-shader Qwen3 decode context. It keeps f16 KV slots on the device and
    /// records only independent output rows as Vulkan workgroups; no cooperative
    /// matrix pipeline or split dot-product reduction is part of this path.
    #[allow(dead_code)]
    pub struct Qwen3DecodeContext<'a> {
        state: Arc<DeviceState>,
        pipelines: Pipelines,
        model: &'a Model,
        layers: Vec<DeviceQwenDecodeLayer>,
        final_norm: Buffer,
        lm_head: DeviceDecodeWeight,
        activations: QwenDecodeActivations,
        /// Lazily allocated on first `verify_batch`; `None` on the single-token
        /// path so the existing decode buffers and pipeline set are unperturbed.
        batch_activations: Option<QwenDecodeBatchActivations>,
        capacity: usize,
        position: usize,
    }

    #[allow(dead_code)]
    impl<'a> Qwen3DecodeContext<'a> {
        pub fn new(
            gemm: VulkanGemm,
            pipeline_cache: Option<PathBuf>,
            model: &'a Model,
            capacity: usize,
        ) -> Result<Self> {
            ensure!(
                matches!(gemm, VulkanGemm::Plain),
                "Qwen3 Vulkan decode is plain-shader only; pass --vulkan-gemm plain"
            );
            ensure!(
                capacity > 0,
                "Qwen3 Vulkan decode cache bucket must be positive"
            );
            ensure!(
                model.config.num_attention_heads % model.config.num_key_value_heads == 0,
                "Qwen3 Vulkan decode has invalid GQA heads"
            );
            let started = Instant::now();
            let state = DeviceState::new(gemm, pipeline_cache)?;
            ensure!(
                !model.weight_quantization.is_quantized() || state.shader_int8_supported,
                "Vulkan GPU lacks shader int8 required for Q8_0 decode"
            );
            let hidden = model.config.hidden_size;
            let query_heads = model.config.num_attention_heads;
            let kv_heads = model.config.num_key_value_heads;
            let head_dim = model.config.head_dim;
            let kv_width = kv_heads * head_dim;
            let layers = model
                .layers
                .iter()
                .map(|layer| {
                    Ok(DeviceQwenDecodeLayer {
                        input_norm: Buffer::from_f32(state.clone(), &layer.input_norm.weight.data)?,
                        post_attention_norm: Buffer::from_f32(
                            state.clone(),
                            &layer.post_attention_norm.weight.data,
                        )?,
                        q_weight: DeviceDecodeWeight::upload(state.clone(), &layer.q_proj)?,
                        q_norm: Buffer::from_f32(state.clone(), &layer.q_norm.weight.data)?,
                        k_weight: DeviceDecodeWeight::upload(state.clone(), &layer.k_proj)?,
                        k_norm: Buffer::from_f32(state.clone(), &layer.k_norm.weight.data)?,
                        v_weight: DeviceDecodeWeight::upload(state.clone(), &layer.v_proj)?,
                        o_weight: DeviceDecodeWeight::upload(state.clone(), &layer.o_proj)?,
                        gate_weight: DeviceDecodeWeight::upload(state.clone(), &layer.gate_proj)?,
                        up_weight: DeviceDecodeWeight::upload(state.clone(), &layer.up_proj)?,
                        down_weight: DeviceDecodeWeight::upload(state.clone(), &layer.down_proj)?,
                        key_cache: Buffer::new(
                            state.clone(),
                            capacity * kv_width * size_of::<u16>(),
                        )?,
                        value_cache: Buffer::new(
                            state.clone(),
                            capacity * kv_width * size_of::<u16>(),
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let lm_head = model.lm_head()?;
            let lm_head = DeviceDecodeWeight::upload_parts(
                state.clone(),
                &lm_head.data,
                model.lm_head_q8_0().map(|quantized| quantized.as_bytes()),
            )?;
            let final_norm = Buffer::from_f32(state.clone(), &model.final_norm.weight.data)?;
            let activations = QwenDecodeActivations::new(
                state.clone(),
                hidden,
                query_heads,
                kv_heads,
                head_dim,
                model.config.intermediate_size,
                model.config.vocab_size,
                capacity,
            )?;
            let pipelines = Pipelines::create(&state)?;
            let (allocations, allocated_bytes) = state.immutable_allocation_summary();
            eprintln!(
                "Vulkan Qwen3 decode persistent weights: upload_ms={:.3} layers={} cache_slots={} storage=f16/q8_0 kv=device-resident allocations={allocations} allocated_bytes={allocated_bytes}",
                started.elapsed().as_secs_f64() * 1_000.0,
                layers.len(),
                capacity,
            );
            Ok(Self {
                state,
                pipelines,
                model,
                layers,
                final_norm,
                lm_head,
                activations,
                batch_activations: None,
                capacity,
                position: 0,
            })
        }

        fn rope(&self, position: usize) -> (Vec<f32>, Vec<f32>) {
            let head_dim = self.model.config.head_dim;
            let mut cosine = Vec::with_capacity(head_dim);
            let mut sine = Vec::with_capacity(head_dim);
            for index in 0..head_dim {
                let rotary_index = index % (head_dim / 2);
                let frequency = 1.0
                    / self
                        .model
                        .config
                        .rope_theta
                        .powf((2 * rotary_index) as f32 / head_dim as f32);
                let (sin, cos) = (position as f32 * frequency).sin_cos();
                cosine.push(cos);
                sine.push(sin);
            }
            (cosine, sine)
        }

        fn embedding(&self, token: u32) -> Result<&[f32]> {
            let token = token as usize;
            ensure!(
                token < self.model.config.vocab_size,
                "token id {token} outside Qwen3 vocabulary"
            );
            let hidden = self.model.config.hidden_size;
            Ok(&self.model.embeddings.data[token * hidden..(token + 1) * hidden])
        }

        #[allow(clippy::too_many_arguments)]
        fn record_matvec(
            recorder: &mut Recorder<'_>,
            pipelines: &Pipelines,
            input: &Buffer,
            weight: &DeviceDecodeWeight,
            output: &Buffer,
            rows: usize,
            columns: usize,
            layer: usize,
            stage: StageClass,
        ) -> Result<()> {
            let params = FourU32 {
                a: rows as u32,
                b: columns as u32,
                c: 0,
                d: 0,
            };
            if let Some(q8_0) = &weight.q8_0 {
                recorder.dispatch(
                    pipelines
                        .decode_matvec_q8_0
                        .context("Vulkan GPU lacks shader int8 required for Q8_0 decode")?,
                    &[input, q8_0, output],
                    &params,
                    // Four lanes in each subgroup own four independent rows;
                    // each active lane still performs its row's full serial dot.
                    [recorder.state.subgroup_groups(rows, 4), 1, 1],
                    layer,
                    stage,
                )
            } else {
                recorder.dispatch(
                    pipelines.decode_matvec,
                    &[
                        input,
                        weight
                            .f16
                            .as_ref()
                            .context("Vulkan decode f16 weight missing")?,
                        output,
                    ],
                    &params,
                    // Wave 4 seam 3: four lanes in each subgroup own four
                    // independent rows, each still performing its row's full
                    // serial f32 dot (mirrors the Q8 pack-four dispatch).
                    [recorder.state.subgroup_groups(rows, 4), 1, 1],
                    layer,
                    stage,
                )
            }
        }

        fn run_token(
            &mut self,
            token: u32,
            position: usize,
            produce_logits: bool,
        ) -> Result<Vec<f32>> {
            ensure!(
                position < self.capacity,
                "Qwen3 Vulkan decode cache capacity exhausted"
            );
            self.activations
                .x0
                .write(&encode_f16_bits(self.embedding(token)?))?;
            let (cosine, sine) = self.rope(position);
            self.activations.cosine.write(&cosine)?;
            self.activations.sine.write(&sine)?;

            let command_pool = unsafe {
                self.state.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.state.queue_family),
                    None,
                )?
            };
            let command = unsafe {
                self.state.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0]
            };
            let max_sets = (self.layers.len() as u32 * 16 + 2).max(2);
            let descriptor_pool = unsafe {
                self.state.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(max_sets)
                        .pool_sizes(&[vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(max_sets * DESCRIPTOR_BINDINGS)]),
                    None,
                )?
            };
            let record_result = (|| {
                unsafe {
                    self.state.device.begin_command_buffer(
                        command,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )?;
                }
                {
                    let mut recorder = Recorder {
                        state: &self.state,
                        command,
                        descriptor_pool,
                        descriptor_sets: Vec::new(),
                        pipelines: &self.pipelines,
                        profile: None,
                    };
                    let hidden = self.model.config.hidden_size;
                    let query_heads = self.model.config.num_attention_heads;
                    let kv_heads = self.model.config.num_key_value_heads;
                    let head_dim = self.model.config.head_dim;
                    let query_width = query_heads * head_dim;
                    let kv_width = kv_heads * head_dim;
                    let intermediate = self.model.config.intermediate_size;
                    for (layer_index, layer) in self.layers.iter().enumerate() {
                        recorder.dispatch(
                            self.pipelines.decode_rms_norm,
                            &[
                                &self.activations.x0,
                                &layer.input_norm,
                                &self.activations.normed,
                            ],
                            &FamilyNormParams {
                                rows: 1,
                                width: hidden as u32,
                                epsilon: self.model.config.rms_norm_eps,
                                identity: 0,
                            },
                            [self.state.subgroup_groups(1, 1), 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                        Self::record_matvec(
                            &mut recorder,
                            &self.pipelines,
                            &self.activations.normed,
                            &layer.q_weight,
                            &self.activations.q_raw,
                            query_width,
                            hidden,
                            layer_index,
                            StageClass::GemmQkv,
                        )?;
                        Self::record_matvec(
                            &mut recorder,
                            &self.pipelines,
                            &self.activations.normed,
                            &layer.k_weight,
                            &self.activations.k_raw,
                            kv_width,
                            hidden,
                            layer_index,
                            StageClass::GemmQkv,
                        )?;
                        Self::record_matvec(
                            &mut recorder,
                            &self.pipelines,
                            &self.activations.normed,
                            &layer.v_weight,
                            &self.activations.v_raw,
                            kv_width,
                            hidden,
                            layer_index,
                            StageClass::GemmQkv,
                        )?;
                        let qk_params = DecodeHeadParams {
                            heads: query_heads as u32,
                            head_dim: head_dim as u32,
                            position: position as u32,
                            cache_stride: query_width as u32,
                            epsilon: self.model.config.rms_norm_eps,
                            write_cache: 0,
                            unused0: 0,
                            unused1: 0,
                        };
                        recorder.dispatch(
                            self.pipelines.decode_head_norm_rope,
                            &[
                                &self.activations.q_raw,
                                &layer.q_norm,
                                &self.activations.cosine,
                                &self.activations.sine,
                                &self.activations.q,
                            ],
                            &qk_params,
                            [query_heads as u32, 1, 1],
                            layer_index,
                            StageClass::LayoutTranspose,
                        )?;
                        recorder.dispatch(
                            self.pipelines.decode_head_norm_rope,
                            &[
                                &self.activations.k_raw,
                                &layer.k_norm,
                                &self.activations.cosine,
                                &self.activations.sine,
                                &layer.key_cache,
                            ],
                            &DecodeHeadParams {
                                heads: kv_heads as u32,
                                head_dim: head_dim as u32,
                                position: position as u32,
                                cache_stride: kv_width as u32,
                                epsilon: self.model.config.rms_norm_eps,
                                write_cache: 1,
                                unused0: 0,
                                unused1: 0,
                            },
                            [kv_heads as u32, 1, 1],
                            layer_index,
                            StageClass::LayoutTranspose,
                        )?;
                        recorder.dispatch(
                            self.pipelines.decode_value_cache,
                            &[&self.activations.v_raw, &layer.value_cache],
                            &FourU32 {
                                a: kv_width as u32,
                                b: position as u32,
                                c: 0,
                                d: 0,
                            },
                            [kv_width.div_ceil(256) as u32, 1, 1],
                            layer_index,
                            StageClass::LayoutTranspose,
                        )?;
                        recorder.dispatch(
                            self.pipelines.decode_attention,
                            &[
                                &self.activations.q,
                                &layer.key_cache,
                                &layer.value_cache,
                                &self.activations.scores,
                                &self.activations.attention,
                            ],
                            &DecodeAttentionParams {
                                position: position as u32,
                                query_heads: query_heads as u32,
                                kv_heads: kv_heads as u32,
                                head_dim: head_dim as u32,
                                capacity: self.capacity as u32,
                                unused0: 0,
                                unused1: 0,
                                unused2: 0,
                            },
                            [query_heads as u32, 1, 1],
                            layer_index,
                            StageClass::SoftmaxMask,
                        )?;
                        Self::record_matvec(
                            &mut recorder,
                            &self.pipelines,
                            &self.activations.attention,
                            &layer.o_weight,
                            &self.activations.projected,
                            hidden,
                            query_width,
                            layer_index,
                            StageClass::GemmOut,
                        )?;
                        recorder.dispatch(
                            self.pipelines.add_residual,
                            &[
                                &self.activations.projected,
                                &self.activations.x0,
                                &self.activations.x1,
                            ],
                            &FourU32 {
                                a: hidden as u32,
                                b: 0,
                                c: 0,
                                d: 0,
                            },
                            [hidden.div_ceil(256) as u32, 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                        recorder.dispatch(
                            self.pipelines.decode_rms_norm,
                            &[
                                &self.activations.x1,
                                &layer.post_attention_norm,
                                &self.activations.normed,
                            ],
                            &FamilyNormParams {
                                rows: 1,
                                width: hidden as u32,
                                epsilon: self.model.config.rms_norm_eps,
                                identity: 0,
                            },
                            [self.state.subgroup_groups(1, 1), 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                        Self::record_matvec(
                            &mut recorder,
                            &self.pipelines,
                            &self.activations.normed,
                            &layer.gate_weight,
                            &self.activations.gate,
                            intermediate,
                            hidden,
                            layer_index,
                            StageClass::GemmMlpUp,
                        )?;
                        Self::record_matvec(
                            &mut recorder,
                            &self.pipelines,
                            &self.activations.normed,
                            &layer.up_weight,
                            &self.activations.up,
                            intermediate,
                            hidden,
                            layer_index,
                            StageClass::GemmMlpUp,
                        )?;
                        recorder.dispatch(
                            self.pipelines.swiglu,
                            &[
                                &self.activations.gate,
                                &self.activations.up,
                                &self.activations.activated,
                            ],
                            &FourU32 {
                                a: intermediate as u32,
                                b: 0,
                                c: 0,
                                d: 0,
                            },
                            [intermediate.div_ceil(256) as u32, 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                        Self::record_matvec(
                            &mut recorder,
                            &self.pipelines,
                            &self.activations.activated,
                            &layer.down_weight,
                            &self.activations.projected,
                            hidden,
                            intermediate,
                            layer_index,
                            StageClass::GemmMlpDown,
                        )?;
                        recorder.dispatch(
                            self.pipelines.add_residual,
                            &[
                                &self.activations.projected,
                                &self.activations.x1,
                                &self.activations.x0,
                            ],
                            &FourU32 {
                                a: hidden as u32,
                                b: 0,
                                c: 0,
                                d: 0,
                            },
                            [hidden.div_ceil(256) as u32, 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                    }
                    if produce_logits {
                        recorder.dispatch(
                            self.pipelines.decode_rms_norm,
                            &[&self.activations.x0, &self.final_norm, &self.activations.x1],
                            &FamilyNormParams {
                                rows: 1,
                                width: hidden as u32,
                                epsilon: self.model.config.rms_norm_eps,
                                identity: 0,
                            },
                            [self.state.subgroup_groups(1, 1), 1, 1],
                            self.layers.len(),
                            StageClass::Pointwise,
                        )?;
                        Self::record_matvec(
                            &mut recorder,
                            &self.pipelines,
                            &self.activations.x1,
                            &self.lm_head,
                            &self.activations.logits,
                            self.model.config.vocab_size,
                            hidden,
                            self.layers.len(),
                            StageClass::GemmMlpDown,
                        )?;
                    }
                }
                unsafe { self.state.device.end_command_buffer(command)? };
                Ok::<(), anyhow::Error>(())
            })();
            if let Err(error) = record_result {
                unsafe {
                    self.state
                        .device
                        .destroy_descriptor_pool(descriptor_pool, None);
                    self.state
                        .device
                        .free_command_buffers(command_pool, &[command]);
                    self.state.device.destroy_command_pool(command_pool, None);
                }
                return Err(error);
            }
            let submit_result = (|| unsafe {
                self.state.device.reset_fences(&[self.state.upload_fence])?;
                self.state.device.queue_submit(
                    self.state.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command])],
                    self.state.upload_fence,
                )?;
                self.state
                    .device
                    .wait_for_fences(&[self.state.upload_fence], true, u64::MAX)?;
                Ok::<(), anyhow::Error>(())
            })();
            let read_result = if produce_logits && submit_result.is_ok() {
                self.activations
                    .logits
                    .read_f32(self.model.config.vocab_size)
            } else {
                Ok(Vec::new())
            };
            unsafe {
                self.state
                    .device
                    .destroy_descriptor_pool(descriptor_pool, None);
                self.state
                    .device
                    .free_command_buffers(command_pool, &[command]);
                self.state.device.destroy_command_pool(command_pool, None);
            }
            submit_result?;
            read_result
        }

        /// Index into the per-K specialized mat-mat pipeline arrays for K in
        /// {1,2,4,8,16} (index 0->1, 1->2, 2->4, 3->8, 4->16). The host rounds
        /// the requested batch up to the next power of two and dispatches only
        /// the first `batch` columns; columns >= batch are computed but not
        /// written, so any batch <= 16 is exact while the common power-of-two
        /// spans use a tight specialization.
        fn batched_pipeline_index(batch: usize) -> usize {
            match batch {
                1 => 0,
                2 => 1,
                3..=4 => 2,
                5..=8 => 3,
                _ => 4,
            }
        }

        /// Per-column RoPE tables for a batched forward. Column k uses the
        /// angle at absolute position `base_position + k`. Returns flattened
        /// `[K][head_dim]` cosine and sine tables.
        fn rope_batch(&self, base_position: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
            let head_dim = self.model.config.head_dim;
            let mut cosine = Vec::with_capacity(k * head_dim);
            let mut sine = Vec::with_capacity(k * head_dim);
            for column in 0..k {
                let position = base_position + column;
                for index in 0..head_dim {
                    let rotary_index = index % (head_dim / 2);
                    let frequency = 1.0
                        / self
                            .model
                            .config
                            .rope_theta
                            .powf((2 * rotary_index) as f32 / head_dim as f32);
                    let (sin, cos) = (position as f32 * frequency).sin_cos();
                    cosine.push(cos);
                    sine.push(sin);
                }
            }
            (cosine, sine)
        }

        /// Record a batched mat-mat dispatch (f16 or Q8) using the per-K
        /// specialized pipeline. The input/output buffers are laid out as
        /// `[K][width]`; the weight is the single-token weight (shared across
        /// all K columns). `rows` is the output width, `columns` is the input
        /// width (the K dimension of the mat-mat).
        #[allow(clippy::too_many_arguments)]
        fn record_matvec_batch(
            recorder: &mut Recorder<'_>,
            pipelines: &Pipelines,
            input: &Buffer,
            weight: &DeviceDecodeWeight,
            output: &Buffer,
            rows: usize,
            columns: usize,
            batch: usize,
            layer: usize,
            stage: StageClass,
        ) -> Result<()> {
            // K=1 uses the single-token matvec shader directly: the batched
            // mat-mat shader's K=1 specialization still wraps the accumulation
            // in a column loop that the glslc compiler optimizes differently
            // than the scalar single-token path on the Q8 shader, breaking the
            // byte-exact gate. The single-token matvec is bit-identical by
            // construction, so K=1 dispatches through it and only K>=2 uses
            // the batched mat-mat kernels.
            if batch == 1 {
                return Self::record_matvec(
                    recorder, pipelines, input, weight, output, rows, columns, layer, stage,
                );
            }
            // Q8 at K>=2: the AMD RDNA3 driver's shader compiler reorders the
            // f32 accumulation when multiple column accumulators are present
            // in the batched mat-mat shader (even with scalar registers and
            // identical per-column operation order), breaking the byte-exact
            // gate. Fall back to K sequential single-token dispatches via the
            // column-offset Q8 matvec — each column uses the exact single-
            // token shader, so the result is bit-identical by construction.
            // The weight streams K times (no sharing); the Q8 K-curve measures
            // the cost of NOT sharing, which is the honest answer to whether
            // the mat-mat shape helps Q8 on RDNA3.
            if let Some(q8_0) = &weight.q8_0 {
                let pipeline = pipelines
                    .batched
                    .matvec_q8_0_column
                    .context("Vulkan GPU lacks shader int8 required for Q8_0 batched decode")?;
                let vecs = columns / 4; // uvec2 elements per column
                for k in 0..batch {
                    let params = FourU32 {
                        a: rows as u32,
                        b: columns as u32,
                        c: (k * vecs) as u32, // input offset (uvec2 elements)
                        d: (k * rows) as u32, // output offset (f32 elements)
                    };
                    recorder.dispatch(
                        pipeline,
                        &[input, q8_0, output],
                        &params,
                        [recorder.state.subgroup_groups(rows, 4), 1, 1],
                        layer,
                        stage,
                    )?;
                }
                return Ok(());
            }
            let params = FourU32 {
                a: rows as u32,
                b: columns as u32,
                c: batch as u32,
                d: 0,
            };
            let index = Self::batched_pipeline_index(batch);
            if let Some(q8_0) = &weight.q8_0 {
                let pipeline = pipelines
                    .batched
                    .matvec_q8_0
                    .as_ref()
                    .context("Vulkan GPU lacks shader int8 required for Q8_0 batched decode")?
                    [index];
                recorder.dispatch(
                    pipeline,
                    &[input, q8_0, output],
                    &params,
                    // Four lanes in each subgroup own four independent rows;
                    // each active lane still performs its row's full serial dot
                    // for all K columns.
                    [recorder.state.subgroup_groups(rows, 4), 1, 1],
                    layer,
                    stage,
                )
            } else {
                let pipeline = pipelines.batched.matvec_f16[index];
                recorder.dispatch(
                    pipeline,
                    &[
                        input,
                        weight
                            .f16
                            .as_ref()
                            .context("Vulkan decode f16 weight missing")?,
                        output,
                    ],
                    &params,
                    [recorder.state.subgroup_groups(rows, 4), 1, 1],
                    layer,
                    stage,
                )
            }
        }

        /// Run K token positions through the transformer in ONE command
        /// submission (mat-mat with K columns), mirroring the Metal
        /// `verify_batch` contract. All K KV slots are written before any
        /// column's attention runs, so column k's causal prefix (positions
        /// <= base_position + k) is fully resident and identical to the
        /// sequential path. Returns the K*vocab f32 logits (row k is the logits
        /// after position base_position + k).
        fn run_batch(&mut self, tokens: &[u32], base_position: usize) -> Result<Vec<f32>> {
            let k = tokens.len();
            ensure!(k >= 1, "batched decode requires at least one token");
            ensure!(
                k <= 16,
                "batched decode supports at most 16 tokens, got {k}"
            );
            ensure!(
                base_position + k <= self.capacity,
                "batched decode exceeds cache capacity"
            );
            ensure!(
                tokens
                    .iter()
                    .all(|&token| (token as usize) < self.model.config.vocab_size),
                "batched decode received a token outside the Qwen3 vocabulary"
            );

            // Lazily allocate the batched activation buffers on first use.
            if self.batch_activations.is_none() {
                self.batch_activations = Some(QwenDecodeBatchActivations::new(
                    self.state.clone(),
                    self.model.config.hidden_size,
                    self.model.config.num_attention_heads,
                    self.model.config.num_key_value_heads,
                    self.model.config.head_dim,
                    self.model.config.intermediate_size,
                    self.model.config.vocab_size,
                    self.capacity,
                    16,
                )?);
            }
            let batch_act = self
                .batch_activations
                .as_ref()
                .expect("batch activations allocated above");

            // Upload the K embedding rows and per-column RoPE tables.
            let hidden = self.model.config.hidden_size;
            let mut input_f16 = Vec::with_capacity(k * hidden);
            for &token in tokens {
                input_f16.extend(encode_f16_bits(self.embedding(token)?));
            }
            batch_act.x0.write(&input_f16)?;
            let (cosine, sine) = self.rope_batch(base_position, k);
            batch_act.cosine.write(&cosine)?;
            batch_act.sine.write(&sine)?;

            let command_pool = unsafe {
                self.state.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.state.queue_family),
                    None,
                )?
            };
            let command = unsafe {
                self.state.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0]
            };
            // Each layer records: RMSNorm, Q/K/V mat-mat, head-norm+RoPE (Q and
            // K), value-cache write, attention, O mat-mat, residual add,
            // post-attention RMSNorm, gate/up mat-mat, SwiGLU, down mat-mat,
            // MLP residual add. Plus final RMSNorm + LM-head mat-mat. The Q8
            // path dispatches K single-token matvecs per mat-mat stage (the
            // column-offset fallback), so the worst case is 6 mat-mat stages
            // * K dispatches + 10 non-mat-mat stages per layer, plus the final
            // RMSNorm and K-dispatch LM-head. Use a generous upper bound to
            // avoid OUT_OF_POOL_MEMORY.
            let q8 = self.lm_head.q8_0.is_some();
            let matmat_stages = 6u32; // Q, K, V, O, gate/up (2), down
            let non_matmat_stages = 10u32;
            let dispatches_per_layer = if q8 {
                matmat_stages * k as u32 + non_matmat_stages
            } else {
                16
            };
            let final_dispatches = if q8 { 1 + k as u32 } else { 2 };
            let max_sets =
                (self.layers.len() as u32 * dispatches_per_layer + final_dispatches).max(2);
            // The Q8 column-offset fallback dispatches K matvecs per mat-mat
            // stage, which can require thousands of descriptor sets for K=16.
            // Use a generous pool size to avoid OUT_OF_POOL_MEMORY.
            let max_sets = max_sets.max(4096);
            let descriptor_pool = unsafe {
                self.state.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(max_sets)
                        .pool_sizes(&[vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(max_sets * DESCRIPTOR_BINDINGS)]),
                    None,
                )?
            };
            let record_result = (|| {
                unsafe {
                    self.state.device.begin_command_buffer(
                        command,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )?;
                }
                {
                    let mut recorder = Recorder {
                        state: &self.state,
                        command,
                        descriptor_pool,
                        descriptor_sets: Vec::new(),
                        pipelines: &self.pipelines,
                        profile: None,
                    };
                    let query_heads = self.model.config.num_attention_heads;
                    let kv_heads = self.model.config.num_key_value_heads;
                    let head_dim = self.model.config.head_dim;
                    let query_width = query_heads * head_dim;
                    let kv_width = kv_heads * head_dim;
                    let intermediate = self.model.config.intermediate_size;
                    for (layer_index, layer) in self.layers.iter().enumerate() {
                        // Input RMSNorm (K columns).
                        recorder.dispatch(
                            self.pipelines.batched.rms_norm,
                            &[&batch_act.x0, &layer.input_norm, &batch_act.normed],
                            &FamilyNormParams {
                                rows: k as u32,
                                width: hidden as u32,
                                epsilon: self.model.config.rms_norm_eps,
                                identity: k as u32,
                            },
                            [self.state.subgroup_groups(k, 1), 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                        // Q/K/V mat-mat (K columns, shared weight).
                        Self::record_matvec_batch(
                            &mut recorder,
                            &self.pipelines,
                            &batch_act.normed,
                            &layer.q_weight,
                            &batch_act.q_raw,
                            query_width,
                            hidden,
                            k,
                            layer_index,
                            StageClass::GemmQkv,
                        )?;
                        Self::record_matvec_batch(
                            &mut recorder,
                            &self.pipelines,
                            &batch_act.normed,
                            &layer.k_weight,
                            &batch_act.k_raw,
                            kv_width,
                            hidden,
                            k,
                            layer_index,
                            StageClass::GemmQkv,
                        )?;
                        Self::record_matvec_batch(
                            &mut recorder,
                            &self.pipelines,
                            &batch_act.normed,
                            &layer.v_weight,
                            &batch_act.v_raw,
                            kv_width,
                            hidden,
                            k,
                            layer_index,
                            StageClass::GemmQkv,
                        )?;
                        // Q head-norm+RoPE (no cache write).
                        recorder.dispatch(
                            self.pipelines.batched.head_norm_rope,
                            &[
                                &batch_act.q_raw,
                                &layer.q_norm,
                                &batch_act.cosine,
                                &batch_act.sine,
                                &batch_act.q,
                            ],
                            &DecodeHeadParams {
                                heads: query_heads as u32,
                                head_dim: head_dim as u32,
                                position: base_position as u32,
                                cache_stride: query_width as u32,
                                epsilon: self.model.config.rms_norm_eps,
                                write_cache: 0,
                                unused0: k as u32,
                                unused1: 0,
                            },
                            [query_heads as u32, k as u32, 1],
                            layer_index,
                            StageClass::LayoutTranspose,
                        )?;
                        // K head-norm+RoPE (writes K cache slots).
                        recorder.dispatch(
                            self.pipelines.batched.head_norm_rope,
                            &[
                                &batch_act.k_raw,
                                &layer.k_norm,
                                &batch_act.cosine,
                                &batch_act.sine,
                                &layer.key_cache,
                            ],
                            &DecodeHeadParams {
                                heads: kv_heads as u32,
                                head_dim: head_dim as u32,
                                position: base_position as u32,
                                cache_stride: kv_width as u32,
                                epsilon: self.model.config.rms_norm_eps,
                                write_cache: 1,
                                unused0: k as u32,
                                unused1: 0,
                            },
                            [kv_heads as u32, k as u32, 1],
                            layer_index,
                            StageClass::LayoutTranspose,
                        )?;
                        // V value-cache write (K slots).
                        recorder.dispatch(
                            self.pipelines.batched.value_cache,
                            &[&batch_act.v_raw, &layer.value_cache],
                            &FourU32 {
                                a: kv_width as u32,
                                b: base_position as u32,
                                c: k as u32,
                                d: 0,
                            },
                            [kv_width.div_ceil(256) as u32, 1, 1],
                            layer_index,
                            StageClass::LayoutTranspose,
                        )?;
                        // Attention (K columns, causal mask inside window).
                        recorder.dispatch(
                            self.pipelines.batched.attention,
                            &[
                                &batch_act.q,
                                &layer.key_cache,
                                &layer.value_cache,
                                &batch_act.scores,
                                &batch_act.attention,
                            ],
                            &DecodeAttentionParams {
                                position: base_position as u32,
                                query_heads: query_heads as u32,
                                kv_heads: kv_heads as u32,
                                head_dim: head_dim as u32,
                                capacity: self.capacity as u32,
                                unused0: k as u32,
                                unused1: 0,
                                unused2: 0,
                            },
                            [query_heads as u32, k as u32, 1],
                            layer_index,
                            StageClass::SoftmaxMask,
                        )?;
                        // O projection mat-mat.
                        Self::record_matvec_batch(
                            &mut recorder,
                            &self.pipelines,
                            &batch_act.attention,
                            &layer.o_weight,
                            &batch_act.projected,
                            hidden,
                            query_width,
                            k,
                            layer_index,
                            StageClass::GemmOut,
                        )?;
                        // Residual add -> x1.
                        recorder.dispatch(
                            self.pipelines.batched.add_residual,
                            &[&batch_act.projected, &batch_act.x0, &batch_act.x1],
                            &FourU32 {
                                a: hidden as u32,
                                b: k as u32,
                                c: 0,
                                d: 0,
                            },
                            [hidden.div_ceil(256) as u32, 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                        // Post-attention RMSNorm.
                        recorder.dispatch(
                            self.pipelines.batched.rms_norm,
                            &[&batch_act.x1, &layer.post_attention_norm, &batch_act.normed],
                            &FamilyNormParams {
                                rows: k as u32,
                                width: hidden as u32,
                                epsilon: self.model.config.rms_norm_eps,
                                identity: k as u32,
                            },
                            [self.state.subgroup_groups(k, 1), 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                        // Gate / Up mat-mat.
                        Self::record_matvec_batch(
                            &mut recorder,
                            &self.pipelines,
                            &batch_act.normed,
                            &layer.gate_weight,
                            &batch_act.gate,
                            intermediate,
                            hidden,
                            k,
                            layer_index,
                            StageClass::GemmMlpUp,
                        )?;
                        Self::record_matvec_batch(
                            &mut recorder,
                            &self.pipelines,
                            &batch_act.normed,
                            &layer.up_weight,
                            &batch_act.up,
                            intermediate,
                            hidden,
                            k,
                            layer_index,
                            StageClass::GemmMlpUp,
                        )?;
                        // SwiGLU.
                        recorder.dispatch(
                            self.pipelines.batched.swiglu,
                            &[&batch_act.gate, &batch_act.up, &batch_act.activated],
                            &FourU32 {
                                a: intermediate as u32,
                                b: k as u32,
                                c: 0,
                                d: 0,
                            },
                            [intermediate.div_ceil(256) as u32, 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                        // Down mat-mat.
                        Self::record_matvec_batch(
                            &mut recorder,
                            &self.pipelines,
                            &batch_act.activated,
                            &layer.down_weight,
                            &batch_act.projected,
                            hidden,
                            intermediate,
                            k,
                            layer_index,
                            StageClass::GemmMlpDown,
                        )?;
                        // MLP residual add -> x0.
                        recorder.dispatch(
                            self.pipelines.batched.add_residual,
                            &[&batch_act.projected, &batch_act.x1, &batch_act.x0],
                            &FourU32 {
                                a: hidden as u32,
                                b: k as u32,
                                c: 0,
                                d: 0,
                            },
                            [hidden.div_ceil(256) as u32, 1, 1],
                            layer_index,
                            StageClass::Pointwise,
                        )?;
                    }
                    // Final RMSNorm + LM-head mat-mat.
                    recorder.dispatch(
                        self.pipelines.batched.rms_norm,
                        &[&batch_act.x0, &self.final_norm, &batch_act.x1],
                        &FamilyNormParams {
                            rows: k as u32,
                            width: hidden as u32,
                            epsilon: self.model.config.rms_norm_eps,
                            identity: k as u32,
                        },
                        [self.state.subgroup_groups(k, 1), 1, 1],
                        self.layers.len(),
                        StageClass::Pointwise,
                    )?;
                    Self::record_matvec_batch(
                        &mut recorder,
                        &self.pipelines,
                        &batch_act.x1,
                        &self.lm_head,
                        &batch_act.logits,
                        self.model.config.vocab_size,
                        hidden,
                        k,
                        self.layers.len(),
                        StageClass::GemmMlpDown,
                    )?;
                }
                unsafe { self.state.device.end_command_buffer(command)? };
                Ok::<(), anyhow::Error>(())
            })();
            if let Err(error) = record_result {
                unsafe {
                    self.state
                        .device
                        .destroy_descriptor_pool(descriptor_pool, None);
                    self.state
                        .device
                        .free_command_buffers(command_pool, &[command]);
                    self.state.device.destroy_command_pool(command_pool, None);
                }
                return Err(error);
            }
            let submit_result = (|| unsafe {
                self.state.device.reset_fences(&[self.state.upload_fence])?;
                self.state.device.queue_submit(
                    self.state.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command])],
                    self.state.upload_fence,
                )?;
                self.state
                    .device
                    .wait_for_fences(&[self.state.upload_fence], true, u64::MAX)?;
                Ok::<(), anyhow::Error>(())
            })();
            let read_result = if submit_result.is_ok() {
                batch_act.logits.read_f32(k * self.model.config.vocab_size)
            } else {
                Ok(Vec::new())
            };
            unsafe {
                self.state
                    .device
                    .destroy_descriptor_pool(descriptor_pool, None);
                self.state
                    .device
                    .free_command_buffers(command_pool, &[command]);
                self.state.device.destroy_command_pool(command_pool, None);
            }
            submit_result?;
            read_result
        }

        /// Batched verification: runs `tokens.len()` positions through one
        /// batched forward and returns the full per-position f32 logits,
        /// flattened as `tokens.len()` contiguous `vocab_size` rows (row `i` is
        /// the logits after position `self.position + i`). By construction each
        /// row is bit-identical to a sequential `advance` at that position.
        pub fn verify_batch_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            let base = self.position;
            // K=1 dispatches through the single-token path: the batched mat-mat
            // shader's K=1 specialization still wraps the accumulation in a
            // column loop that the glslc compiler optimizes differently than
            // the scalar single-token path on the Q8 shader, breaking the
            // byte-exact gate. The single-token `run_token` is bit-identical by
            // construction, so K=1 uses it directly and only K>=2 uses the
            // batched mat-mat kernels.
            if tokens.len() == 1 {
                let logits = self.run_token(tokens[0], base, true)?;
                self.position = base + 1;
                return Ok(logits);
            }
            let logits = self.run_batch(tokens, base)?;
            self.position = base + tokens.len();
            Ok(logits)
        }

        pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            ensure!(!tokens.is_empty(), "decode prompt must not be empty");
            ensure!(
                tokens.len() <= self.capacity,
                "decode prompt exceeds cache bucket"
            );
            self.position = 0;
            let mut logits = Vec::new();
            for (index, &token) in tokens.iter().enumerate() {
                logits = self.run_token(token, index, index + 1 == tokens.len())?;
            }
            self.position = tokens.len();
            Ok(logits)
        }

        pub fn advance(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
            ensure!(
                position == self.position,
                "Qwen3 Vulkan decode cache position mismatch: requested {position}, current {}",
                self.position
            );
            let logits = self.run_token(token, position, true)?;
            self.position += 1;
            Ok(logits)
        }

        /// Sets the logical cache position for speculative-decode rewind. KV
        /// data is addressed by position; attention reads only positions <= the
        /// logical length, and the next forward overwrites later slots.
        pub fn set_position(&mut self, position: usize) {
            self.position = position;
        }

        pub fn inspect_cache_layer(&self, layer: usize) -> Result<Vec<f32>> {
            let layer = self
                .layers
                .get(layer)
                .context("Qwen3 Vulkan cache layer out of range")?;
            let values_per_cache =
                self.capacity * self.model.config.num_key_value_heads * self.model.config.head_dim;
            let mut bits = layer.key_cache.read_u16(values_per_cache)?;
            bits.extend(layer.value_cache.read_u16(values_per_cache)?);
            Ok(decode_f16_bits(&bits))
        }
    }

    impl Drop for Qwen3DecodeContext<'_> {
        fn drop(&mut self) {
            unsafe {
                let _ = self.state.device.device_wait_idle();
                for pipeline in self.pipelines.all() {
                    self.state.device.destroy_pipeline(pipeline, None);
                }
            }
        }
    }

    unsafe fn bytes_of<T: Copy>(value: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts((value as *const T).cast(), size_of::<T>()) }
    }
}

#[cfg(not(feature = "vulkan"))]
mod enabled {
    use std::path::PathBuf;

    use anyhow::{bail, Result};

    use super::super::{EncoderLayer, VulkanGemm};
    use super::{ModernBertLayer, Qwen3Layer};
    use crate::qwen3::Model;

    pub struct VulkanContext;

    impl VulkanContext {
        pub fn new(_gemm: VulkanGemm, _pipeline_cache_path: Option<PathBuf>) -> Result<Self> {
            bail!("Vulkan provider requires cargo feature `vulkan`")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encoder_forward(
            &mut self,
            _hidden_states: &mut [f32],
            _attention_mask: &[u8],
            _batch: usize,
            _seq: usize,
            _hidden: usize,
            _heads: usize,
            _intermediate: usize,
            _layer_norm_eps: f32,
            _layers: &[EncoderLayer],
        ) -> Result<Vec<Vec<f32>>> {
            bail!("Vulkan provider requires cargo feature `vulkan`")
        }
    }

    pub struct ModernBertContext;

    impl ModernBertContext {
        pub fn new(_gemm: VulkanGemm, _pipeline_cache: Option<PathBuf>) -> Result<Self> {
            bail!("Vulkan provider requires cargo feature `vulkan`")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &mut self,
            _hidden_states: &mut [f32],
            _attention_mask: &[u8],
            _batch: usize,
            _seq: usize,
            _hidden: usize,
            _heads: usize,
            _intermediate: usize,
            _epsilon: f32,
            _global_theta: f32,
            _local_theta: f32,
            _half_window: usize,
            _layers: &[ModernBertLayer<'_>],
            _final_norm: &[f32],
        ) -> Result<()> {
            bail!("Vulkan provider requires cargo feature `vulkan`")
        }
    }

    #[allow(dead_code)]
    pub struct Qwen3DecodeContext<'a> {
        _model: std::marker::PhantomData<&'a Model>,
    }

    #[allow(dead_code)]
    impl<'a> Qwen3DecodeContext<'a> {
        pub fn new(
            _gemm: VulkanGemm,
            _pipeline_cache: Option<PathBuf>,
            _model: &'a Model,
            _capacity: usize,
        ) -> Result<Self> {
            bail!("Vulkan Qwen3 decode requires cargo feature `vulkan`")
        }

        pub fn prefill(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
            bail!("Vulkan Qwen3 decode requires cargo feature `vulkan`")
        }

        pub fn advance(&mut self, _token: u32, _position: usize) -> Result<Vec<f32>> {
            bail!("Vulkan Qwen3 decode requires cargo feature `vulkan`")
        }

        pub fn inspect_cache_layer(&self, _layer: usize) -> Result<Vec<f32>> {
            bail!("Vulkan Qwen3 decode requires cargo feature `vulkan`")
        }
    }

    pub struct Qwen3Context;

    impl Qwen3Context {
        pub fn new(_gemm: VulkanGemm, _pipeline_cache: Option<PathBuf>) -> Result<Self> {
            bail!("Vulkan provider requires cargo feature `vulkan`")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &mut self,
            _hidden_states: &mut [f32],
            _attention_mask: &[u8],
            _batch: usize,
            _seq: usize,
            _hidden: usize,
            _query_heads: usize,
            _kv_heads: usize,
            _head_dim: usize,
            _intermediate: usize,
            _epsilon: f32,
            _theta: f32,
            _layers: &[Qwen3Layer<'_>],
            _final_norm: &[f32],
        ) -> Result<()> {
            bail!("Vulkan provider requires cargo feature `vulkan`")
        }
    }
}

#[cfg_attr(not(feature = "vulkan"), allow(unused_imports))]
pub use enabled::{ModernBertContext, Qwen3Context, Qwen3DecodeContext, VulkanContext};
