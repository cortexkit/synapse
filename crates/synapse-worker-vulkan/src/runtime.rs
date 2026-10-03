use std::{collections::BTreeMap, ffi::CStr, io::Cursor, sync::Arc};

use anyhow::{anyhow, ensure, Context, Result};
use ash::{vk, Entry};
use synapse_parity::{
    manifest::{Family, Model, Pooling, Profile},
    vulkan::{arena_bytes, buffer_plan, Buffer as PlannedBuffer, FLOOR_SEQUENCES},
};

use crate::{
    admission::{Adapter, Required},
    Padded, SHADERS,
};

struct Instance {
    _entry: Entry,
    raw: ash::Instance,
    api_version: u32,
}
impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.raw.destroy_instance(None);
        }
    }
}
fn instance() -> Result<Instance> {
    let library = if cfg!(windows) {
        "vulkan-1.dll"
    } else {
        "libvulkan.so.1"
    };
    instance_from_loader(library, false)
}

fn instance_from_loader(
    library: impl AsRef<std::ffi::OsStr>,
    portability: bool,
) -> Result<Instance> {
    let entry = unsafe { Entry::load_from(library) }.context("vulkan_no_device")?;
    let loader_version = unsafe { entry.try_enumerate_instance_version() }
        .context("vulkan_no_device")?
        .unwrap_or(vk::API_VERSION_1_0);
    let api_version = if loader_version >= vk::API_VERSION_1_3 {
        vk::API_VERSION_1_3
    } else {
        vk::API_VERSION_1_2
    };
    let app = vk::ApplicationInfo::default().api_version(api_version);
    let extensions = [ash::khr::portability_enumeration::NAME.as_ptr()];
    let mut info = vk::InstanceCreateInfo::default().application_info(&app);
    if portability {
        info = info
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
            .enabled_extension_names(&extensions);
    }
    let raw = unsafe { entry.create_instance(&info, None) }.context("vulkan_no_device")?;
    Ok(Instance {
        _entry: entry,
        raw,
        api_version,
    })
}

fn inspect(instance: &Instance, physical: vk::PhysicalDevice, index: usize) -> Adapter {
    unsafe {
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
        instance
            .raw
            .get_physical_device_properties2(physical, &mut props);
        let p = props.properties;
        let mut float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut float16)
            .push_next(&mut storage16);
        instance
            .raw
            .get_physical_device_features2(physical, &mut features);
        let memory = instance.raw.get_physical_device_memory_properties(physical);
        let extensions = instance
            .raw
            .enumerate_device_extension_properties(physical)
            .unwrap_or_default();
        let cooperative_matrix = extensions.iter().any(|e| {
            CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::cooperative_matrix::NAME
        });
        Adapter {
            index,
            discrete: p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
            cpu: p.device_type == vk::PhysicalDeviceType::CPU,
            vendor: p.vendor_id,
            api_major: vk::api_version_major(p.api_version),
            api_minor: vk::api_version_minor(p.api_version),
            shader_float16: float16.shader_float16 == vk::TRUE,
            storage_buffer16_bit_access: storage16.storage_buffer16_bit_access == vk::TRUE,
            subgroup_arithmetic: subgroup
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::ARITHMETIC),
            subgroup_compute_stage: subgroup
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE),
            max_storage_buffer_range: u64::from(p.limits.max_storage_buffer_range),
            device_local_heaps: memory.memory_heaps[..memory.memory_heap_count as usize]
                .iter()
                .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|h| h.size)
                .collect(),
            cooperative_matrix,
        }
    }
}

pub fn enumerate() -> Result<Vec<Adapter>, String> {
    let instance = instance().map_err(|_| "vulkan_no_device".to_owned())?;
    let physical = unsafe { instance.raw.enumerate_physical_devices() }
        .map_err(|_| "vulkan_no_device".to_owned())?;
    Ok(physical
        .into_iter()
        .enumerate()
        .map(|(i, p)| inspect(&instance, p, i))
        .collect())
}

pub struct Prepared {
    pub adapter: Adapter,
    device: Arc<Device>,
}

pub fn prepare(required: Required) -> Result<Prepared> {
    let device = Device::new(required)?;
    Ok(Prepared {
        adapter: device.adapter.clone(),
        device,
    })
}

/// Explicit development-only Apple GPU entry point. Release LOAD and probe
/// retain their loader names, vendor policy and cooperative selection.
#[cfg(all(target_os = "macos", feature = "moltenvk-diagnostic"))]
pub fn prepare_moltenvk_parity(loader: &std::path::Path, required: Required) -> Result<Prepared> {
    let instance = instance_from_loader(loader.as_os_str(), true)?;
    let device = Device::with_instance(required, instance, true)?;
    eprintln!(
        "MoltenVK plain-path adapter: {}",
        serde_json::to_string(&device.adapter)?
    );
    Ok(Prepared {
        adapter: device.adapter.clone(),
        device,
    })
}

/// Production Vulkan builds cannot import the diagnostic entry point.
/// ```compile_fail
/// use synapse_worker_vulkan::runtime::prepare;
/// use synapse_worker_vulkan::runtime::prepare_moltenvk_parity;
/// ```
#[cfg(not(feature = "moltenvk-diagnostic"))]
pub struct DiagnosticApiAbsent;

struct Device {
    instance: Instance,
    adapter: Adapter,
    raw: ash::Device,
    physical: vk::PhysicalDevice,
    queue: vk::Queue,
    pool: vk::CommandPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    plain: vk::Pipeline,
    cooperative: Option<vk::Pipeline>,
    heap_index: u32,
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            if let Some(p) = self.cooperative {
                self.raw.destroy_pipeline(p, None);
            }
            self.raw.destroy_pipeline(self.plain, None);
            self.raw.destroy_pipeline_layout(self.pipeline_layout, None);
            self.raw.destroy_descriptor_pool(self.descriptor_pool, None);
            self.raw
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.raw.destroy_command_pool(self.pool, None);
            self.raw.destroy_device(None);
        }
    }
}

impl Device {
    fn new(required: Required) -> Result<Arc<Self>> {
        Self::with_instance(required, instance()?, false)
    }

    fn with_instance(
        required: Required,
        instance: Instance,
        diagnostic: bool,
    ) -> Result<Arc<Self>> {
        let physicals =
            unsafe { instance.raw.enumerate_physical_devices() }.context("vulkan_no_device")?;
        let adapters: Vec<Adapter> = physicals
            .iter()
            .enumerate()
            .map(|(index, physical)| inspect(&instance, *physical, index))
            .collect();
        #[cfg(all(target_os = "macos", feature = "moltenvk-diagnostic"))]
        let adapters = if diagnostic {
            let diagnostics: Vec<_> = adapters
                .into_iter()
                .filter(|a| a.vendor == 0x106b)
                .collect();
            for adapter in &diagnostics {
                let mut floor_check = adapter.clone();
                floor_check.vendor = 0x1002;
                floor_check.check(required).map_err(|code| anyhow!(code))?;
            }
            diagnostics
        } else {
            adapters
        };
        let adapter = if diagnostic {
            adapters
                .into_iter()
                .next()
                .context("no Apple diagnostic adapter")?
        } else {
            crate::admission::select(adapters, required).map_err(|code| anyhow!(code))?
        };
        let physical = physicals[adapter.index];
        let queues = unsafe {
            instance
                .raw
                .get_physical_device_queue_family_properties(physical)
        };
        let family = queues
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .context("no compute queue")? as u32;
        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut float16 =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
        let mut storage16 =
            vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);
        let can_cooperate =
            !diagnostic && instance.api_version >= vk::API_VERSION_1_3 && adapter.use_cooperative();
        let mut cooperative = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut memory_model = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
        if can_cooperate {
            let mut query = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut cooperative)
                .push_next(&mut memory_model);
            unsafe {
                instance
                    .raw
                    .get_physical_device_features2(physical, &mut query);
            }
        }
        let coop_shape = if can_cooperate {
            let extension =
                ash::khr::cooperative_matrix::Instance::new(&instance._entry, &instance.raw);
            unsafe { extension.get_physical_device_cooperative_matrix_properties(physical) }
                .unwrap_or_default()
                .iter()
                .any(|p| {
                    p.m_size == 16
                        && p.n_size == 16
                        && p.k_size == 16
                        && p.scope == vk::ScopeKHR::SUBGROUP
                        && p.a_type == vk::ComponentTypeKHR::FLOAT16
                        && p.b_type == vk::ComponentTypeKHR::FLOAT16
                        && p.c_type == vk::ComponentTypeKHR::FLOAT32
                        && p.result_type == vk::ComponentTypeKHR::FLOAT32
                })
        } else {
            false
        };
        let coop_enabled = can_cooperate
            && cooperative.cooperative_matrix == vk::TRUE
            && memory_model.vulkan_memory_model == vk::TRUE
            && coop_shape;
        cooperative = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(coop_enabled);
        memory_model = vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
            .vulkan_memory_model(coop_enabled);
        let mut extensions: Vec<_> = if coop_enabled {
            vec![ash::khr::cooperative_matrix::NAME.as_ptr()]
        } else {
            vec![]
        };
        if diagnostic {
            extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&extensions)
            .push_next(&mut float16)
            .push_next(&mut storage16);
        if coop_enabled {
            info = info
                .push_next(&mut cooperative)
                .push_next(&mut memory_model);
        }
        let raw = unsafe { instance.raw.create_device(physical, &info, None) }?;
        let queue = unsafe { raw.get_device_queue(family, 0) };
        let memory = unsafe { instance.raw.get_physical_device_memory_properties(physical) };
        let heap_index = memory.memory_heaps[..memory.memory_heap_count as usize]
            .iter()
            .enumerate()
            .filter(|(_, h)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .max_by_key(|(_, h)| h.size)
            .context("no device-local heap")?
            .0 as u32;
        let mut device = Self {
            instance,
            adapter,
            raw,
            physical,
            queue,
            heap_index,
            pool: vk::CommandPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            plain: vk::Pipeline::null(),
            cooperative: None,
        };
        let raw = &device.raw;
        let pool = unsafe {
            raw.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )
        }?;
        device.pool = pool;
        let bindings: Vec<_> = (0..7)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let descriptor_layout = unsafe {
            raw.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;
        device.descriptor_layout = descriptor_layout;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(7)];
        let descriptor_pool = unsafe {
            raw.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }?;
        device.descriptor_pool = descriptor_pool;
        let layouts = [descriptor_layout];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(64)];
        let pipeline_layout = unsafe {
            raw.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
        }?;
        device.pipeline_layout = pipeline_layout;
        let make_pipeline = |name: &str| -> Result<vk::Pipeline> {
            let bytes = SHADERS
                .iter()
                .find(|(n, _)| *n == name)
                .context("missing embedded shader")?
                .1;
            let code = ash::util::read_spv(&mut Cursor::new(bytes))?;
            let module = unsafe {
                raw.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            }?;
            let entry = c"main";
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);
            let result =
                unsafe { raw.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
            unsafe {
                raw.destroy_shader_module(module, None);
            }
            result.map(|p| p[0]).map_err(|(pipelines, e)| {
                for pipeline in pipelines {
                    unsafe {
                        raw.destroy_pipeline(pipeline, None);
                    }
                }
                anyhow!("create compute pipeline: {e}")
            })
        };
        device.plain = make_pipeline("plain")?;
        device.cooperative = if coop_enabled {
            Some(make_pipeline("cooperative")?)
        } else {
            None
        };
        Ok(Arc::new(device))
    }

    fn submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { self.raw.allocate_command_buffers(&info) }?[0];
        unsafe {
            self.raw.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        record(command);
        unsafe {
            self.raw.end_command_buffer(command)?;
            let commands = [command];
            let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
            self.raw
                .queue_submit(self.queue, &submits, vk::Fence::null())?;
            self.raw.queue_wait_idle(self.queue)?;
            self.raw.free_command_buffers(self.pool, &[command]);
        }
        Ok(())
    }
}

struct TransferBuffer {
    device: Arc<Device>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}
impl Drop for TransferBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

struct Arena {
    device: Arc<Device>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    slices: BTreeMap<String, (u64, u64)>,
    requested_bytes: u64,
    driver_bytes: u64,
    mapped: bool,
}
impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}
impl Arena {
    fn new(device: Arc<Device>, plan: &[PlannedBuffer], cap: u64) -> Result<Self> {
        let layout = crate::allocation::Layout::new(plan, cap).map_err(|e| anyhow!(e))?;
        let requested_bytes = layout.requested_bytes;
        let slices = layout.slices;
        let info = vk::BufferCreateInfo::default()
            .size(requested_bytes)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.raw.create_buffer(&info, None) }?;
        let requirements = unsafe { device.raw.get_buffer_memory_requirements(buffer) };
        let mut arena = Self {
            device: device.clone(),
            buffer,
            memory: vk::DeviceMemory::null(),
            slices,
            requested_bytes,
            driver_bytes: requirements.size,
            mapped: false,
        };
        let props = unsafe {
            device
                .instance
                .raw
                .get_physical_device_memory_properties(device.physical)
        };
        let candidates: Vec<_> = props.memory_types[..props.memory_type_count as usize]
            .iter()
            .enumerate()
            .filter(|(i, m)| {
                requirements.memory_type_bits & (1 << i) != 0
                    && m.heap_index == device.heap_index
                    && m.property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .collect();
        let (memory_type, memory_properties) = candidates
            .iter()
            .copied()
            .max_by_key(|(_, m)| {
                m.property_flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .context("vulkan_insufficient_memory")?;
        let mapped = memory_properties.property_flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let memory_type = memory_type as u32;
        let memory = unsafe {
            device.raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }?;
        arena.memory = memory;
        arena.mapped = mapped;
        unsafe {
            device.raw.bind_buffer_memory(buffer, memory, 0)?;
        }
        Ok(arena)
    }

    fn descriptor(&self, name: &str) -> Result<vk::DescriptorBufferInfo> {
        let &(offset, range) = self
            .slices
            .get(name)
            .with_context(|| format!("unknown buffer {name}"))?;
        Ok(vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(offset)
            .range(range))
    }
    fn transfer(&self, name: &str, destination: &mut [u8], upload: bool) -> Result<()> {
        let &(offset, capacity) = self.slices.get(name).context("unknown buffer")?;
        ensure!(
            destination.len() as u64 <= capacity,
            "buffer capacity exceeded"
        );
        let original_len = destination.len();
        let mut padded = destination.to_vec();
        padded.resize(original_len.div_ceil(4) * 4, 0);
        let bytes = padded.as_mut_slice();
        let device = &self.device;
        if self.mapped {
            unsafe {
                let pointer = device.raw.map_memory(
                    self.memory,
                    0,
                    self.driver_bytes,
                    vk::MemoryMapFlags::empty(),
                )?;
                let pointer = pointer.cast::<u8>().add(offset as usize);
                if upload {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len());
                } else {
                    std::ptr::copy_nonoverlapping(pointer, destination.as_mut_ptr(), original_len);
                }
                device.raw.unmap_memory(self.memory);
            }
            return Ok(());
        }
        let info = vk::BufferCreateInfo::default()
            .size(bytes.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
        let buffer = unsafe { device.raw.create_buffer(&info, None) }?;
        let req = unsafe { device.raw.get_buffer_memory_requirements(buffer) };
        let mut staging = TransferBuffer {
            device: device.clone(),
            buffer,
            memory: vk::DeviceMemory::null(),
        };
        let props = unsafe {
            device
                .instance
                .raw
                .get_physical_device_memory_properties(device.physical)
        };
        let index = props.memory_types[..props.memory_type_count as usize]
            .iter()
            .enumerate()
            .find(|(i, m)| {
                req.memory_type_bits & (1 << i) != 0
                    && !m
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && m.property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .context("no host transfer memory")?
            .0 as u32;
        let memory = unsafe {
            device.raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(index),
                None,
            )
        }?;
        staging.memory = memory;
        unsafe {
            device.raw.bind_buffer_memory(buffer, memory, 0)?;
        }
        if upload {
            unsafe {
                let pointer =
                    device
                        .raw
                        .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.cast(), bytes.len());
                device.raw.unmap_memory(memory);
            }
        }
        device.submit(|command| unsafe {
            let (source, target, source_offset, target_offset) = if upload {
                (buffer, self.buffer, 0, offset)
            } else {
                (self.buffer, buffer, offset, 0)
            };
            device.raw.cmd_copy_buffer(
                command,
                source,
                target,
                &[vk::BufferCopy::default()
                    .src_offset(source_offset)
                    .dst_offset(target_offset)
                    .size(bytes.len() as u64)],
            );
        })?;
        if !upload {
            unsafe {
                let pointer =
                    device
                        .raw
                        .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())?;
                std::ptr::copy_nonoverlapping(pointer.cast(), bytes.as_mut_ptr(), bytes.len());
                device.raw.unmap_memory(memory);
            }
        }
        drop(staging);
        if !upload {
            destination.copy_from_slice(&padded[..original_len]);
        }
        Ok(())
    }

    fn dispatch(&self, params: Params, buffers: [&str; 7], count: u32) -> Result<()> {
        let device = &self.device;
        let descriptors: Vec<_> = buffers
            .iter()
            .map(|name| self.descriptor(name))
            .collect::<Result<_>>()?;
        let layouts = [device.descriptor_layout];
        unsafe {
            device.raw.reset_descriptor_pool(
                device.descriptor_pool,
                vk::DescriptorPoolResetFlags::empty(),
            )?;
        }
        let set = unsafe {
            device.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(device.descriptor_pool)
                    .set_layouts(&layouts),
            )
        }?[0];
        let writes: Vec<_> = descriptors
            .iter()
            .enumerate()
            .map(|(i, d)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(d))
            })
            .collect();
        unsafe {
            device.raw.update_descriptor_sets(&writes, &[]);
        }
        let cooperative = params.op == 1
            && params.inner % 16 == 0
            && params.cols % 16 == 0
            && device.cooperative.is_some();
        let groups = if cooperative {
            params.rows.div_ceil(16) * params.cols.div_ceil(16)
        } else {
            count.div_ceil(64)
        };
        let words = params.words();
        let bytes: Vec<_> = words.into_iter().flat_map(u32::to_ne_bytes).collect();
        device.submit(|command| unsafe {
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            device.raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
            device.raw.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                if cooperative {
                    device.cooperative.unwrap()
                } else {
                    device.plain
                },
            );
            device.raw.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                device.pipeline_layout,
                0,
                &[set],
                &[],
            );
            device.raw.cmd_push_constants(
                command,
                device.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &bytes,
            );
            device
                .raw
                .cmd_dispatch(command, groups.min(65535), groups.div_ceil(65535), 1);
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)];
            device.raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        })
    }
}

#[derive(Clone, Copy, Default)]
struct Params {
    op: u32,
    rows: u32,
    cols: u32,
    inner: u32,
    seq: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    flags: u32,
    offset: u32,
    stride: u32,
    window: u32,
    eps: f32,
    theta: f32,
    scale: f32,
    unused: f32,
}
impl Params {
    fn words(self) -> [u32; 16] {
        [
            self.op,
            self.rows,
            self.cols,
            self.inner,
            self.seq,
            self.heads,
            self.kv_heads,
            self.dim,
            self.flags,
            self.offset,
            self.stride,
            self.window,
            self.eps.to_bits(),
            self.theta.to_bits(),
            self.scale.to_bits(),
            self.unused.to_bits(),
        ]
    }
}

pub struct Engine {
    arena: Arena,
    model: Model,
    profile: Profile,
    manifest_context: u32,
}
impl Engine {
    pub fn load(
        model: Model,
        profile: Profile,
        prepared: Prepared,
        required: Required,
        header: &synapse_parity::safetensors::Header,
        data: &[u8],
    ) -> Result<Self> {
        let context = crate::manifest().admission.max_context_tokens;
        let plan = buffer_plan(
            &model,
            profile.storage_dtype,
            &profile.fp32_tensors,
            u64::from(context),
            FLOOR_SEQUENCES,
            u64::from(
                profile
                    .vulkan_sub_batch_max_tokens
                    .context("missing sub-batch ceiling")?,
            ),
        )?;
        ensure!(
            arena_bytes(&plan) <= required.min_device_local_bytes,
            "vulkan_insufficient_memory"
        );
        let mut arena = Arena::new(prepared.device, &plan, required.min_device_local_bytes)?;
        if model.architecture.family == Family::Qwen3 {
            let t = u64::from(profile.vulkan_sub_batch_max_tokens.unwrap());
            let d = model.architecture.int("head_dim")?;
            let q = t * model.architecture.int("num_attention_heads")? * d * 4;
            let kv = t * model.architecture.int("num_key_value_heads")? * d * 4;
            let base = arena.slices["activation:qkv"].0;
            arena.slices.insert("qkv:q".into(), (base, q));
            arena.slices.insert("qkv:k".into(), (base + q, kv));
            arena.slices.insert("qkv:v".into(), (base + q + kv, kv));
        }
        for (name, tensor) in &header.tensors {
            let mut bytes = data[tensor.data_offsets.0..tensor.data_offsets.1].to_vec();
            arena.transfer(&format!("weight:{name}"), &mut bytes, true)?;
        }
        eprintln!(
            "{}",
            serde_json::json!({"vulkan_load_report": {"requested_device_local_bytes":arena.requested_bytes,"driver_memory_requirements_size":arena.driver_bytes,"allocations":1,"heap_index":arena.device.heap_index}})
        );
        Ok(Self {
            arena,
            model,
            profile,
            manifest_context: context,
        })
    }

    fn weight(&self, name: &str) -> String {
        format!("weight:{}{name}", self.model.tensor_prefix)
    }
    fn run(
        &self,
        p: Params,
        input: &str,
        weight: &str,
        output: &str,
        other: &str,
        values: &str,
        count: u32,
    ) -> Result<()> {
        self.arena.dispatch(
            p,
            [
                input,
                weight,
                output,
                other,
                "activation:token_ids",
                "activation:positions",
                values,
            ],
            count,
        )
    }
    fn linear(
        &self,
        rows: u32,
        inner: u32,
        cols: u32,
        input: &str,
        weight: &str,
        output: &str,
        residual: bool,
    ) -> Result<()> {
        self.run(
            Params {
                op: 1,
                rows,
                cols,
                inner,
                flags: u32::from(residual),
                ..Default::default()
            },
            input,
            weight,
            output,
            "activation:hidden",
            input,
            rows * cols,
        )
    }
    fn norm(
        &self,
        rows: u32,
        width: u32,
        input: &str,
        output: &str,
        weight: &str,
        rms: bool,
        eps: f32,
    ) -> Result<()> {
        self.run(
            Params {
                op: 2,
                rows,
                cols: width,
                flags: u32::from(rms),
                eps,
                ..Default::default()
            },
            input,
            weight,
            output,
            input,
            input,
            rows,
        )
    }

    pub fn infer(&self, sequences: &[Vec<i32>]) -> Result<Vec<f32>> {
        ensure!(
            !sequences.is_empty() && sequences.len() <= FLOOR_SEQUENCES as usize,
            "invalid_batch"
        );
        let longest = sequences.iter().map(Vec::len).max().unwrap();
        ensure!(
            longest <= self.manifest_context as usize,
            "sequence_too_long"
        );
        ensure!(
            longest <= self.profile.vulkan_sub_batch_max_tokens.unwrap() as usize,
            "sequence_too_long"
        );
        let padded =
            crate::pad(sequences, self.model.grammar.pad.id as i32).map_err(|e| anyhow!(e))?;
        let mut result = Vec::new();
        // The arena is sized for one maximum-context sequence. Reuse it without
        // changing the request's batch-longest padding or splitting a sequence.
        for (index, &length) in padded.lengths.iter().enumerate() {
            let range = index * padded.width..(index + 1) * padded.width;
            let sub = Padded {
                ids: padded.ids[range.clone()].to_vec(),
                mask: padded.mask[range].to_vec(),
                lengths: vec![length],
                width: padded.width,
            };
            result.extend(self.forward(&sub)?);
        }
        Ok(result)
    }

    fn forward(&self, padded: &Padded) -> Result<Vec<f32>> {
        let arch = &self.model.architecture;
        let h = arch.int("hidden_size")? as u32;
        let heads = arch.int("num_attention_heads")? as u32;
        let intermediate = arch.int("intermediate_size")? as u32;
        let modern = arch.family == Family::Modernbert;
        let kv_heads = if modern {
            heads
        } else {
            arch.int("num_key_value_heads")? as u32
        };
        let dim = if modern {
            h / heads
        } else {
            arch.int("head_dim")? as u32
        };
        ensure!(dim <= 256, "unsupported head dimension");
        let q_width = heads * dim;
        let kv_width = kv_heads * dim;
        let seq = padded.width as u32;
        let batch = padded.lengths.len() as u32;
        let rows = seq * batch;
        let eps = crate::norm_epsilon(&self.model)?;
        let vocab = arch.int("vocab_size")? as i32;
        ensure!(
            padded.ids.iter().all(|id| *id >= 0 && *id < vocab),
            "invalid_tokens"
        );
        self.arena.transfer(
            "activation:token_ids",
            &mut padded
                .ids
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
            true,
        )?;
        self.arena.transfer(
            "activation:positions",
            &mut padded
                .mask
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
            true,
        )?;
        let embedding = self.weight(if modern {
            "embeddings.tok_embeddings.weight"
        } else {
            "embed_tokens.weight"
        });
        let dummy = embedding.as_str();
        self.run(
            Params {
                op: 0,
                rows,
                cols: h,
                ..Default::default()
            },
            "activation:hidden",
            dummy,
            "activation:hidden",
            "activation:normed",
            "activation:normed",
            rows * h,
        )?;
        if modern {
            self.norm(
                rows,
                h,
                "activation:hidden",
                "activation:normed",
                &self.weight("embeddings.norm.weight"),
                false,
                eps,
            )?;
            // The first layer's residual must be the normalized embeddings.
            self.copy("activation:normed", "activation:hidden", rows * h, dummy)?;
        }
        for layer in 0..arch.int("num_hidden_layers")? {
            let prefix = format!("layers.{layer}");
            if modern {
                if layer > 0 {
                    self.norm(
                        rows,
                        h,
                        "activation:hidden",
                        "activation:normed",
                        &self.weight(&format!("{prefix}.attn_norm.weight")),
                        false,
                        eps,
                    )?;
                } else {
                    self.copy("activation:hidden", "activation:normed", rows * h, dummy)?;
                }
                self.linear(
                    rows,
                    h,
                    3 * h,
                    "activation:normed",
                    &self.weight(&format!("{prefix}.attn.Wqkv.weight")),
                    "activation:qkv",
                    false,
                )?;
            } else {
                self.norm(
                    rows,
                    h,
                    "activation:hidden",
                    "activation:normed",
                    &self.weight(&format!("{prefix}.input_layernorm.weight")),
                    true,
                    eps,
                )?;
                self.linear(
                    rows,
                    h,
                    q_width,
                    "activation:normed",
                    &self.weight(&format!("{prefix}.self_attn.q_proj.weight")),
                    "qkv:q",
                    false,
                )?;
                self.linear(
                    rows,
                    h,
                    kv_width,
                    "activation:normed",
                    &self.weight(&format!("{prefix}.self_attn.k_proj.weight")),
                    "qkv:k",
                    false,
                )?;
                self.linear(
                    rows,
                    h,
                    kv_width,
                    "activation:normed",
                    &self.weight(&format!("{prefix}.self_attn.v_proj.weight")),
                    "qkv:v",
                    false,
                )?;
                self.norm(
                    rows * heads,
                    dim,
                    "qkv:q",
                    "activation:attention_out",
                    &self.weight(&format!("{prefix}.self_attn.q_norm.weight")),
                    true,
                    eps,
                )?;
                self.norm(
                    rows * kv_heads,
                    dim,
                    "qkv:k",
                    "activation:mlp_in",
                    &self.weight(&format!("{prefix}.self_attn.k_norm.weight")),
                    true,
                    eps,
                )?;
            }
            let global = !modern || layer % arch.int("global_attn_every_n_layers")? == 0;
            let theta = arch.float(if !modern {
                "rope_theta"
            } else if global {
                "global_rope_theta"
            } else {
                "local_rope_theta"
            })? as f32;
            let stride = if modern { 3 * h } else { q_width };
            self.run(
                Params {
                    op: 5,
                    rows,
                    cols: q_width,
                    seq,
                    dim,
                    stride,
                    theta,
                    ..Default::default()
                },
                if modern {
                    "activation:qkv"
                } else {
                    "activation:attention_out"
                },
                dummy,
                if modern {
                    "activation:attention_out"
                } else {
                    "qkv:q"
                },
                "activation:hidden",
                "activation:hidden",
                rows * q_width,
            )?;
            self.run(
                Params {
                    op: 5,
                    rows,
                    cols: kv_width,
                    seq,
                    dim,
                    stride: if modern { 3 * h } else { kv_width },
                    offset: if modern { h } else { 0 },
                    theta,
                    ..Default::default()
                },
                if modern {
                    "activation:qkv"
                } else {
                    "activation:mlp_in"
                },
                dummy,
                if modern { "activation:mlp_in" } else { "qkv:k" },
                "activation:hidden",
                "activation:hidden",
                rows * kv_width,
            )?;
            let query = if modern {
                "activation:attention_out"
            } else {
                "qkv:q"
            };
            let attention_output = if modern {
                "activation:normed"
            } else {
                "activation:attention_out"
            };
            self.run(
                Params {
                    op: 6,
                    rows,
                    seq,
                    heads,
                    kv_heads,
                    dim,
                    flags: u32::from(!modern),
                    stride: if modern { 3 * h } else { kv_width },
                    offset: if modern { 2 * h } else { 0 },
                    window: if modern && !global {
                        arch.int("local_attention")? as u32 / 2
                    } else {
                        0
                    },
                    ..Default::default()
                },
                query,
                dummy,
                attention_output,
                if modern { "activation:mlp_in" } else { "qkv:k" },
                if modern { "activation:qkv" } else { "qkv:v" },
                rows * heads,
            )?;
            self.linear(
                rows,
                q_width,
                h,
                attention_output,
                &self.weight(&format!(
                    "{prefix}.{}",
                    if modern {
                        "attn.Wo.weight"
                    } else {
                        "self_attn.o_proj.weight"
                    }
                )),
                "activation:mlp_in",
                true,
            )?;
            self.copy("activation:mlp_in", "activation:hidden", rows * h, dummy)?;
            self.norm(
                rows,
                h,
                "activation:hidden",
                "activation:normed",
                &self.weight(&format!(
                    "{prefix}.{}",
                    if modern {
                        "mlp_norm.weight"
                    } else {
                        "post_attention_layernorm.weight"
                    }
                )),
                !modern,
                eps,
            )?;
            if modern {
                self.linear(
                    rows,
                    h,
                    2 * intermediate,
                    "activation:normed",
                    &self.weight(&format!("{prefix}.mlp.Wi.weight")),
                    "activation:mlp_in",
                    false,
                )?;
                self.run(
                    Params {
                        op: 4,
                        rows,
                        cols: intermediate,
                        stride: 2 * intermediate,
                        ..Default::default()
                    },
                    "activation:mlp_in",
                    dummy,
                    "activation:mlp_act",
                    "activation:hidden",
                    "activation:hidden",
                    rows * intermediate,
                )?;
            } else {
                self.linear(
                    rows,
                    h,
                    intermediate,
                    "activation:normed",
                    &self.weight(&format!("{prefix}.mlp.gate_proj.weight")),
                    "activation:mlp_in",
                    false,
                )?;
                self.linear(
                    rows,
                    h,
                    intermediate,
                    "activation:normed",
                    &self.weight(&format!("{prefix}.mlp.up_proj.weight")),
                    "activation:mlp_act",
                    false,
                )?;
                self.run(
                    Params {
                        op: 4,
                        rows,
                        cols: intermediate,
                        stride: intermediate,
                        flags: 1,
                        ..Default::default()
                    },
                    "activation:mlp_in",
                    dummy,
                    "activation:mlp_in",
                    "activation:mlp_act",
                    "activation:hidden",
                    rows * intermediate,
                )?;
            }
            self.linear(
                rows,
                intermediate,
                h,
                if modern {
                    "activation:mlp_act"
                } else {
                    "activation:mlp_in"
                },
                &self.weight(&format!(
                    "{prefix}.mlp.{}",
                    if modern {
                        "Wo.weight"
                    } else {
                        "down_proj.weight"
                    }
                )),
                "activation:normed",
                true,
            )?;
            self.copy("activation:normed", "activation:hidden", rows * h, dummy)?;
        }
        self.norm(
            rows,
            h,
            "activation:hidden",
            "activation:normed",
            &self.weight(if modern {
                "final_norm.weight"
            } else {
                "norm.weight"
            }),
            !modern,
            eps,
        )?;
        let pooling = match self.model.grammar.pooling {
            Pooling::Cls => 0,
            Pooling::MaskedMean => 1,
            Pooling::LastNonPad => 2,
        };
        self.run(
            Params {
                op: 7,
                rows: batch,
                cols: h,
                seq,
                flags: pooling,
                ..Default::default()
            },
            "activation:normed",
            dummy,
            "activation:pooled",
            "activation:hidden",
            "activation:hidden",
            batch * h,
        )?;
        let output_dim = self.model.output.dimension;
        match self.model.grammar.readout.kind {
            synapse_parity::manifest::ReadoutKind::PooledHiddenState => self.run(
                Params {
                    op: 10,
                    rows: batch,
                    cols: h,
                    ..Default::default()
                },
                "activation:pooled",
                dummy,
                "result",
                "activation:hidden",
                "activation:hidden",
                batch,
            )?,
            synapse_parity::manifest::ReadoutKind::YesNoTwoWaySoftmax => {
                let readout = &self.model.grammar.readout;
                let weight = format!(
                    "weight:{}",
                    readout.weight.as_ref().context("missing readout")?
                );
                for (column, token) in [
                    readout.yes.as_ref().context("missing yes id")?,
                    readout.no.as_ref().context("missing no id")?,
                ]
                .iter()
                .enumerate()
                {
                    // Project one readout row at a time; no full-vocabulary normalization is performed.
                    self.run(
                        Params {
                            op: 1,
                            rows: batch,
                            cols: 1,
                            inner: h,
                            offset: token.id,
                            ..Default::default()
                        },
                        "activation:pooled",
                        &weight,
                        if column == 0 {
                            "activation:head_scratch"
                        } else {
                            "activation:attention_stats"
                        },
                        "activation:hidden",
                        "activation:hidden",
                        batch,
                    )?;
                }
                self.run(
                    Params {
                        op: 12,
                        rows: batch,
                        ..Default::default()
                    },
                    "activation:head_scratch",
                    dummy,
                    "result",
                    "activation:attention_stats",
                    "activation:hidden",
                    batch,
                )?;
            }
            synapse_parity::manifest::ReadoutKind::SigmoidClassifierLogit => {
                let head = self
                    .model
                    .head
                    .as_ref()
                    .context("missing classifier head")?;
                let key = |role: &str| -> Result<String> {
                    Ok(format!(
                        "weight:{}",
                        head.tensors
                            .get(role)
                            .with_context(|| format!("missing head {role}"))?
                            .key
                    ))
                };
                self.linear(
                    batch,
                    h,
                    h,
                    "activation:pooled",
                    &key("dense")?,
                    "activation:head_scratch",
                    false,
                )?;
                self.run(
                    Params {
                        op: 8,
                        rows: batch,
                        cols: h,
                        ..Default::default()
                    },
                    "activation:head_scratch",
                    dummy,
                    "activation:pooled",
                    "activation:hidden",
                    "activation:hidden",
                    batch * h,
                )?;
                self.norm(
                    batch,
                    h,
                    "activation:pooled",
                    "activation:head_scratch",
                    &key("norm")?,
                    false,
                    head.norm_eps.context("missing head norm epsilon")? as f32,
                )?;
                self.linear(
                    batch,
                    h,
                    1,
                    "activation:head_scratch",
                    &key("classifier")?,
                    "activation:pooled",
                    false,
                )?;
                self.run(
                    Params {
                        op: 13,
                        rows: batch,
                        ..Default::default()
                    },
                    "activation:pooled",
                    &key("classifier_bias")?,
                    "result",
                    "activation:hidden",
                    "activation:hidden",
                    batch,
                )?;
            }
        }
        let mut bytes = vec![0; (batch * output_dim * 4) as usize];
        self.arena.transfer("result", &mut bytes, false)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn copy(&self, input: &str, output: &str, count: u32, dummy: &str) -> Result<()> {
        self.run(
            Params {
                op: 11,
                rows: count,
                ..Default::default()
            },
            input,
            dummy,
            output,
            input,
            input,
            count,
        )
    }
}
