use std::{
    collections::BTreeMap,
    ffi::CStr,
    io::Cursor,
    sync::{
        atomic::{AtomicBool, Ordering::Relaxed},
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::{anyhow, ensure, Context, Result};
use ash::{vk, Entry};
use synapse_parity::{
    manifest::{Family, Model, Pooling, Profile},
    vulkan::{arena_bytes, buffer_plan, Buffer as PlannedBuffer, FLOOR_SEQUENCES},
};

use crate::{
    admission::{Adapter, Required},
    driver_failure::{allocation_failure, QueueWaitFailed},
    Padded, SHADERS,
};

#[path = "speech_batch_policy.rs"]
mod speech_batch_policy;
use speech_batch_policy::{SpeechBatchPolicy, SPEECH_BATCH_CAPACITY};

#[cfg(test)]
#[path = "transcribe_timing.rs"]
pub(crate) mod speech_timing;

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
    let device = Device::with_instance(required, instance, AdapterChoice::AppleDiagnostic)?;
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

#[path = "transcribe_decode.rs"]
mod speech_decode;
#[path = "transcribe_runtime.rs"]
pub mod transcribe;

/// Which physical device `Device::with_instance` opens.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AdapterChoice {
    /// The adapter chosen by the production vendor, feature and memory policy.
    Admitted,
    /// The Apple adapter of the explicit MoltenVK diagnostic build.
    #[cfg_attr(
        not(all(target_os = "macos", feature = "moltenvk-diagnostic")),
        allow(dead_code)
    )]
    AppleDiagnostic,
    /// The first CPU (software) adapter, such as lavapipe, so that unit tests
    /// can run shaders on machines without a GPU. Never used outside tests.
    #[cfg(test)]
    Software,
}

/// Creates a compute pipeline from one of the embedded SPIR-V shaders.
fn compute_pipeline(
    raw: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    name: &str,
) -> Result<vk::Pipeline> {
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
    let result = unsafe { raw.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
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
}

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
    attention: vk::Pipeline,
    speech: vk::Pipeline,
    speech_gemm: vk::Pipeline,
    heap_index: u32,
    #[cfg(test)]
    timestamp_valid_bits: u32,
    retirement_unknown: AtomicBool,
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            if let Some(p) = self.cooperative {
                self.raw.destroy_pipeline(p, None);
            }
            self.raw.destroy_pipeline(self.plain, None);
            self.raw.destroy_pipeline(self.attention, None);
            self.raw.destroy_pipeline(self.speech, None);
            self.raw.destroy_pipeline(self.speech_gemm, None);
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
    fn retirement_unknown(&self) -> bool {
        self.retirement_unknown.load(Relaxed)
    }

    fn new(required: Required) -> Result<Arc<Self>> {
        Self::with_instance(required, instance()?, AdapterChoice::Admitted)
    }

    /// A device on the first software adapter, bypassing admission.
    #[cfg(test)]
    fn software_for_tests() -> Result<Arc<Self>> {
        Self::with_instance(
            Required {
                min_storage_buffer_range: 0,
                min_device_local_bytes: 0,
            },
            instance()?,
            AdapterChoice::Software,
        )
    }

    fn with_instance(
        required: Required,
        instance: Instance,
        choice: AdapterChoice,
    ) -> Result<Arc<Self>> {
        let diagnostic = choice == AdapterChoice::AppleDiagnostic;
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
        let adapter = match choice {
            AdapterChoice::AppleDiagnostic => adapters
                .into_iter()
                .next()
                .context("no Apple diagnostic adapter")?,
            #[cfg(test)]
            AdapterChoice::Software => adapters
                .into_iter()
                .find(|a| a.cpu)
                .context("no software Vulkan adapter")?,
            AdapterChoice::Admitted => {
                crate::admission::select(adapters, required).map_err(|code| anyhow!(code))?
            }
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
            #[cfg(test)]
            timestamp_valid_bits: queues[family as usize].timestamp_valid_bits,
            pool: vk::CommandPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            plain: vk::Pipeline::null(),
            cooperative: None,
            attention: vk::Pipeline::null(),
            speech: vk::Pipeline::null(),
            speech_gemm: vk::Pipeline::null(),
            retirement_unknown: AtomicBool::new(false),
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
        let make_pipeline = |name: &str| compute_pipeline(raw, pipeline_layout, name);
        device.plain = make_pipeline("plain")?;
        device.attention = make_pipeline("attention")?;
        device.speech = make_pipeline("transcribe")?;
        device.speech_gemm = make_pipeline("transcribe_gemm")?;
        device.cooperative = if coop_enabled {
            Some(make_pipeline("cooperative")?)
        } else {
            None
        };
        Ok(Arc::new(device))
    }

    fn submit(self: &Arc<Self>, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        self.submit_timed(record, false).map(|_| ())
    }

    fn submit_timed(
        self: &Arc<Self>,
        record: impl FnOnce(vk::CommandBuffer),
        measure: bool,
    ) -> Result<Duration> {
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
        let elapsed = unsafe {
            self.raw.end_command_buffer(command)?;
            let commands = [command];
            let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
            let started = measure.then(Instant::now);
            let submitted = self
                .raw
                .queue_submit(self.queue, &submits, vk::Fence::null());
            // Vulkan treats DEVICE_LOST like success for resource lifetime, so
            // even a submit that returns it needs a wait before resources can drop.
            let waited = if submitted.is_ok() || submitted == Err(vk::Result::ERROR_DEVICE_LOST) {
                self.raw.queue_wait_idle(self.queue)
            } else {
                Ok(())
            };
            if let Err(error) = waited {
                if error == vk::Result::ERROR_DEVICE_LOST {
                    self.raw.free_command_buffers(self.pool, &[command]);
                } else if !self.retirement_unknown.swap(true, Relaxed) {
                    // The queue may still be using these objects. Keep them
                    // and the Vulkan loader alive until the worker exits.
                    std::mem::forget(Arc::clone(self));
                }
                return Err(QueueWaitFailed(error).into());
            }
            let elapsed = started.map_or(Duration::ZERO, |started| started.elapsed());
            self.raw.free_command_buffers(self.pool, &[command]);
            submitted?;
            elapsed
        };
        Ok(elapsed)
    }
}

struct TransferBuffer {
    device: Arc<Device>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}
impl Drop for TransferBuffer {
    fn drop(&mut self) {
        if self.device.retirement_unknown() {
            return;
        }
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
        if self.device.retirement_unknown() {
            return;
        }
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
        }
        .map_err(allocation_failure)?;
        arena.memory = memory;
        arena.mapped = mapped;
        unsafe {
            device.raw.bind_buffer_memory(buffer, memory, 0)?;
        }
        #[cfg(test)]
        speech_timing::report_arena(&arena, memory_type, &props);
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
        #[cfg(test)]
        if req.size >= speech_timing::LARGE_BUFFER {
            speech_timing::report_allocation(&props, name, "staging", req.size, index);
        }
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
        let cooperative = params.op == 1
            && params.inner.is_multiple_of(16)
            && params.cols.is_multiple_of(16)
            && device.cooperative.is_some();
        let pipeline = if params.op == 100 {
            device.speech_gemm
        } else if params.op >= 100 {
            device.speech
        } else if params.op == ATTENTION_OP {
            device.attention
        } else if cooperative {
            device.cooperative.unwrap()
        } else {
            device.plain
        };
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
        let groups = if params.op >= 100 {
            speech_groups(&params, count)
        } else if cooperative {
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
            device
                .raw
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
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

/// Speech GEMM flag: add the bias bound at binding 4.
const SPEECH_GEMM_BIAS: u32 = 1;
/// Speech GEMM flag: compute three projections of one input in one dispatch,
/// with weights at bindings 1, 3 and 4 and outputs at bindings 2, 5 and 6.
const SPEECH_GEMM_THREE: u32 = 2;

fn speech_gemm_groups(params: &Params) -> u32 {
    let matrices = if params.flags & SPEECH_GEMM_THREE != 0 {
        3
    } else {
        1
    };
    params.rows.div_ceil(32) * params.cols.div_ceil(32) * matrices
}

/// Workgroups for a speech operation over `count` elements.
fn speech_groups(params: &Params, count: u32) -> u32 {
    match params.op {
        100 => speech_gemm_groups(params),
        // LayerNorm reduces each row with its own 64-thread workgroup.
        101 => params.rows,
        _ => count.div_ceil(64),
    }
}

struct SpeechDispatch {
    params: Params,
    descriptors: Vec<vk::DescriptorBufferInfo>,
    groups: u32,
    #[cfg(test)]
    label: Option<String>,
}

/// Keeps descriptor sets immutable until the complete layer has finished. The
/// embedding/reranking path retains its independent single-dispatch pool.
struct SpeechBatch {
    device: Arc<Device>,
    pool: vk::DescriptorPool,
    pending: Vec<SpeechDispatch>,
    policy: SpeechBatchPolicy,
    #[cfg(test)]
    dispatch_limit_override: Option<usize>,
    #[cfg(test)]
    stats: speech_batch_policy::SubmissionStats,
    #[cfg(test)]
    timing: Option<speech_timing::Timing>,
}

impl SpeechBatch {
    fn new(device: Arc<Device>) -> Result<Self> {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(7 * SPEECH_BATCH_CAPACITY as u32)];
        let pool = unsafe {
            device.raw.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(SPEECH_BATCH_CAPACITY as u32)
                    .pool_sizes(&sizes),
                None,
            )
        }?;
        Ok(Self {
            device,
            pool,
            pending: Vec::with_capacity(SPEECH_BATCH_CAPACITY),
            policy: SpeechBatchPolicy::default(),
            #[cfg(test)]
            dispatch_limit_override: None,
            #[cfg(test)]
            stats: Default::default(),
            #[cfg(test)]
            timing: None,
        })
    }

    fn dispatch_limit(&self) -> usize {
        #[cfg(test)]
        if let Some(limit) = self.dispatch_limit_override {
            return limit;
        }
        self.policy.dispatch_limit()
    }

    fn push(
        &mut self,
        arena: &Arena,
        params: Params,
        buffers: [&str; 7],
        count: u32,
    ) -> Result<()> {
        ensure!(params.op >= 100, "non-speech op in speech batch");
        let descriptors = buffers
            .iter()
            .map(|name| arena.descriptor(name))
            .collect::<Result<_>>()?;
        let groups = speech_groups(&params, count);
        self.pending.push(SpeechDispatch {
            params,
            descriptors,
            groups,
            #[cfg(test)]
            label: self.timing.as_ref().map(|_| {
                let weight = buffers[1];
                if weight == "dummy" {
                    buffers[2]
                } else {
                    weight
                }
                .to_owned()
            }),
        });
        if self.pending.len() >= self.dispatch_limit() {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let dispatches = std::mem::take(&mut self.pending);
        #[cfg(test)]
        let started = self.timing.as_mut().map(|t| t.begin_group(&dispatches));
        let device = &self.device;
        // The preceding flush waited for completion. No pending command buffer
        // references sets from this pool when it is reset.
        unsafe {
            device
                .raw
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
        }?;
        let layouts = vec![device.descriptor_layout; dispatches.len()];
        let sets = unsafe {
            device.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&layouts),
            )
        }?;
        for (dispatch, &set) in dispatches.iter().zip(&sets) {
            let writes: Vec<_> = dispatch
                .descriptors
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
            unsafe { device.raw.update_descriptor_sets(&writes, &[]) };
        }
        let elapsed = device.submit_timed(
            |command| unsafe {
                #[cfg(test)]
                if let Some(timing) = &self.timing {
                    timing.record_start(command);
                }
                // Make uploads visible once. Between dispatches, compute-to-compute
                // ordering covers read-after-write and write-after-write dependencies,
                // and keeps scratch writes from overtaking preceding reads.
                let before = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                device.raw.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &before,
                    &[],
                    &[],
                );
                for (index, (dispatch, &set)) in dispatches.iter().zip(&sets).enumerate() {
                    if index > 0 {
                        let dependency = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )];
                        device.raw.cmd_pipeline_barrier(
                            command,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &dependency,
                            &[],
                            &[],
                        );
                    }
                    let pipeline = if dispatch.params.op == 100 {
                        device.speech_gemm
                    } else {
                        device.speech
                    };
                    device
                        .raw
                        .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
                    device.raw.cmd_bind_descriptor_sets(
                        command,
                        vk::PipelineBindPoint::COMPUTE,
                        device.pipeline_layout,
                        0,
                        &[set],
                        &[],
                    );
                    let bytes: Vec<_> = dispatch
                        .params
                        .words()
                        .into_iter()
                        .flat_map(u32::to_ne_bytes)
                        .collect();
                    device.raw.cmd_push_constants(
                        command,
                        device.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &bytes,
                    );
                    device.raw.cmd_dispatch(
                        command,
                        dispatch.groups.min(65535),
                        dispatch.groups.div_ceil(65535),
                        1,
                    );
                }
                let after = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)];
                device.raw.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &after,
                    &[],
                    &[],
                );
                #[cfg(test)]
                if let Some(timing) = &self.timing {
                    timing.record_end(command);
                }
            },
            true,
        )?;
        #[cfg(test)]
        if let Some((timing, started)) = self.timing.as_mut().zip(started) {
            timing.finish_group(started, elapsed)?;
        }
        #[cfg(test)]
        self.stats.record(dispatches.len(), elapsed);
        #[cfg(test)]
        if self.dispatch_limit_override.is_some() {
            return Ok(());
        }
        self.policy.completed(dispatches.len(), elapsed);
        Ok(())
    }
}

impl Drop for SpeechBatch {
    fn drop(&mut self) {
        if self.device.retirement_unknown() {
            return;
        }
        unsafe { self.device.raw.destroy_descriptor_pool(self.pool, None) };
    }
}

/// Push-constant operation code for self-attention. Only the attention
/// pipeline (shaders/attention.comp) implements it; it is kept out of the
/// plain shader because its shared-memory tiles would lower the occupancy of
/// every other kernel there.
const ATTENTION_OP: u32 = 6;

/// Threads that share one query in the attention shader, which pads the head
/// to this many 32-channel slices. Two is the floor so that a workgroup never
/// holds more than 32 queries and its score scratch stays within 2048 floats.
fn attention_lanes(dim: u32) -> u32 {
    dim.div_ceil(32).next_power_of_two().max(2)
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
    // Operands name shader descriptor bindings directly; keep their order visible.
    #[allow(clippy::too_many_arguments)]
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
    // Matrix geometry and buffer operands remain explicit at projection call sites.
    #[allow(clippy::too_many_arguments)]
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
    // Keep the normalization geometry, buffer operands and numeric policy explicit.
    #[allow(clippy::too_many_arguments)]
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
        self.infer_progress(sequences, &mut || Ok(()))
    }

    pub fn infer_progress(
        &self,
        sequences: &[Vec<i32>],
        progress: &mut impl FnMut() -> Result<()>,
    ) -> Result<Vec<f32>> {
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
        let mut result = Vec::new();
        // The arena holds one sequence at a time. Run each row at its own width
        // to avoid projecting and attending to padding from longer rows.
        crate::for_each_inference_row(sequences, self.model.grammar()?.pad.id as i32, |row| {
            result.extend(self.forward(row, progress)?);
            progress()?;
            Ok(())
        })?;
        Ok(result)
    }

    fn forward(
        &self,
        padded: &Padded,
        progress: &mut impl FnMut() -> Result<()>,
    ) -> Result<Vec<f32>> {
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
        progress()?;
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
            let lanes = attention_lanes(dim);
            let workgroups = batch * heads * seq.div_ceil(64 / lanes);
            self.run(
                Params {
                    op: ATTENTION_OP,
                    rows,
                    inner: lanes,
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
                workgroups * 64,
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
            // Each dispatch above waits for queue idle. This layer's GPU work
            // has completed before reporting progress; submission alone is not enough.
            progress()?;
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
        let grammar = self.model.grammar()?;
        let pooling = match grammar.pooling {
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
        let output_dim = self.model.output()?.dimension;
        match grammar.readout.kind {
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
                let readout = &grammar.readout;
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
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

#[cfg(test)]
mod tests {
    use super::attention_lanes;
    use super::{speech_gemm_groups, speech_groups, Params, SPEECH_GEMM_BIAS, SPEECH_GEMM_THREE};

    #[test]
    fn transcribe_gemm_register_tiles_cover_ragged_rows_columns_and_offsets() {
        let source = include_str!("../shaders/transcribe_gemm.comp");
        assert!(source.contains("const uint DEPTH=32;"));
        assert!(source.contains("const uint STRIDE=9;"));
        assert!(source.contains("shared vec4 tile_x[DEPTH*STRIDE];"));
        assert!(source.contains("shared vec4 tile_w[DEPTH*STRIDE];"));
        // Rows below the column count select column-major workgroup order;
        // larger row counts select row-major order. Both orders are covered,
        // with and without the fused three-projection mode.
        for (rows, cols, inner, offset, flags) in [
            (35, 37, 65, 3, SPEECH_GEMM_BIAS),
            (70, 33, 40, 1, SPEECH_GEMM_BIAS),
            (35, 37, 65, 0, 0),
            (35, 37, 65, 0, SPEECH_GEMM_THREE),
            (100, 20, 96, 0, SPEECH_GEMM_THREE),
        ] {
            let params = Params {
                op: 100,
                rows,
                cols,
                inner,
                offset,
                flags,
                ..Default::default()
            };
            let (rows, cols, inner, offset) = (
                rows as usize,
                cols as usize,
                inner as usize,
                offset as usize,
            );
            let mut random = Lcg((rows * cols + inner) as u64);
            let x = random.values(rows * inner);
            let weights = [0, 1, 2].map(|_| random.values(cols * inner));
            let bias = random.values(cols);
            let bias = if flags & SPEECH_GEMM_THREE != 0 {
                // The third weight occupies the bias binding.
                weights[2].clone()
            } else {
                bias
            };
            let matrices = if flags & SPEECH_GEMM_THREE != 0 { 3 } else { 1 };
            let (outputs, written) =
                emulate_speech_gemm(&params, &x, [&weights[0], &weights[1], &bias], &bias[..]);
            for (matrix, (output, written)) in outputs.iter().zip(&written).enumerate() {
                if matrix >= matrices {
                    assert!(written.iter().all(|&n| n == 0));
                    continue;
                }
                assert!(output[..offset * cols].iter().all(|v| v.is_nan()));
                assert!(written.iter().all(|&n| n == 1), "{rows}x{cols} {flags}");
                let w = if matrix == 2 { &bias } else { &weights[matrix] };
                for row in 0..rows {
                    for col in 0..cols {
                        let products = (0..inner).map(|k| (x[row * inner + k], w[col * inner + k]));
                        // Ascending k from zero, then the bias: the order of
                        // the scalar shader that preceded the tiled one.
                        let sequential = products.clone().fold(0f32, |sum, (a, b)| sum + a * b);
                        let exact: f64 = products
                            .clone()
                            .map(|(a, b)| f64::from(a) * f64::from(b))
                            .sum();
                        let magnitude: f64 = products
                            .map(|(a, b)| (f64::from(a) * f64::from(b)).abs())
                            .sum();
                        let added = if flags & SPEECH_GEMM_BIAS != 0 {
                            bias[col]
                        } else {
                            0.
                        };
                        let actual = output[(row + offset) * cols + col];
                        assert_eq!(actual.to_bits(), (sequential + added).to_bits());
                        let reference = exact + f64::from(added);
                        // Sequential f32 rounding is bounded by inner * eps * magnitude.
                        let tolerance = 1e-5 * (magnitude + f64::from(added.abs())).max(1.);
                        assert!((f64::from(actual) - reference).abs() <= tolerance);
                    }
                }
            }
        }
    }

    /// Deterministic values in [-1, 1) with fractional bits, so that a
    /// different summation order would change the rounded result.
    struct Lcg(u64);
    impl Lcg {
        fn values(&mut self, count: usize) -> Vec<f32> {
            (0..count)
                .map(|_| {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.
                })
                .collect()
        }
    }

    /// Runs transcribe_gemm.comp's algorithm on the CPU, workgroup by workgroup
    /// and thread by thread, with the same group order, loads, shared-memory
    /// layout and multiply-add order. Returns the three output bindings
    /// (NaN where unwritten) and how often each output element was written.
    fn emulate_speech_gemm(
        params: &Params,
        x: &[f32],
        weights: [&[f32]; 3],
        bias: &[f32],
    ) -> ([Vec<f32>; 3], [Vec<u32>; 3]) {
        const DEPTH: usize = 32;
        const STRIDE: usize = 9;
        let (rows, cols, inner) = (
            params.rows as usize,
            params.cols as usize,
            params.inner as usize,
        );
        let offset = params.offset as usize;
        let matrices = if params.flags & SPEECH_GEMM_THREE != 0 {
            3
        } else {
            1
        };
        let mut outputs = [0, 1, 2].map(|_| vec![f32::NAN; (rows + offset) * cols]);
        let mut written = [0, 1, 2].map(|_| vec![0u32; rows * cols]);
        let (row_tiles, column_tiles) = (rows.div_ceil(32), cols.div_ceil(32));
        let groups = speech_gemm_groups(params) as usize;
        assert_eq!(groups, row_tiles * column_tiles * matrices);
        for group in 0..groups.min(65535) * groups.div_ceil(65535) {
            if group >= groups {
                continue;
            }
            let (row_tile, column_tile) = if rows < cols * matrices {
                (group % row_tiles, group / row_tiles)
            } else {
                (
                    group / (column_tiles * matrices),
                    group % (column_tiles * matrices),
                )
            };
            let matrix = column_tile / column_tiles;
            let (row0, col0) = (row_tile * 32, column_tile % column_tiles * 32);
            let fetch = |start: usize, lane: usize| {
                let (k, first) = (start + lane % 32, lane / 32 * 16);
                let (mut input, mut weight) = ([0f32; 16], [0f32; 16]);
                for (j, (a, b)) in input.iter_mut().zip(&mut weight).enumerate() {
                    let (r, c) = (row0 + first + j, col0 + first + j);
                    if k < inner && r < rows {
                        *a = x[r * inner + k];
                    }
                    if k < inner && c < cols {
                        *b = weights[matrix][c * inner + k];
                    }
                }
                (input, weight)
            };
            let mut next: Vec<_> = (0..64).map(|lane| fetch(0, lane)).collect();
            let mut sums = [[[0f32; 4]; 4]; 64];
            for start in (0..inner).step_by(DEPTH) {
                // NaN marks shared slots that no thread staged.
                let mut tile_x = [[f32::NAN; 4]; DEPTH * STRIDE];
                let mut tile_w = [[f32::NAN; 4]; DEPTH * STRIDE];
                for (lane, (input, weight)) in next.iter().enumerate() {
                    let (k, base) = (lane % 32, lane / 32 * 4);
                    for g in 0..4 {
                        tile_x[k * STRIDE + base + g] = input[4 * g..4 * g + 4].try_into().unwrap();
                        tile_w[k * STRIDE + base + g] =
                            weight[4 * g..4 * g + 4].try_into().unwrap();
                    }
                }
                if start + DEPTH < inner {
                    next = (0..64).map(|lane| fetch(start + DEPTH, lane)).collect();
                }
                for (lane, sum) in sums.iter_mut().enumerate() {
                    for k in 0..DEPTH {
                        let inputs = tile_x[k * STRIDE + lane / 8];
                        let weights = tile_w[k * STRIDE + lane % 8];
                        for (r, row_sum) in sum.iter_mut().enumerate() {
                            for (c, value) in row_sum.iter_mut().enumerate() {
                                *value += inputs[r] * weights[c];
                            }
                        }
                    }
                }
            }
            for (lane, sum) in sums.iter().enumerate() {
                for (r, row_sum) in sum.iter().enumerate() {
                    for (c, value) in row_sum.iter().enumerate() {
                        let (rr, cc) = (row0 + lane / 8 * 4 + r, col0 + lane % 8 * 4 + c);
                        if rr < rows && cc < cols {
                            let added = if params.flags & SPEECH_GEMM_BIAS != 0 {
                                bias[cc]
                            } else {
                                0.
                            };
                            outputs[matrix][(rr + offset) * cols + cc] = value + added;
                            written[matrix][rr * cols + cc] += 1;
                        }
                    }
                }
            }
        }
        (outputs, written)
    }

    /// Mirrors LayerNorm in transcribe.comp: one 64-thread workgroup per row,
    /// strided per-thread sums, then the fixed pairwise tree.
    fn emulate_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], cols: usize) -> Vec<f32> {
        let workgroup_sum = |partial: &mut [f32; 64]| {
            let mut half_width = 32;
            while half_width > 0 {
                let (low, high) = partial.split_at_mut(half_width);
                for (a, b) in low.iter_mut().zip(&high[..half_width]) {
                    *a += *b;
                }
                half_width >>= 1;
            }
            partial[0]
        };
        let mut output = vec![f32::NAN; x.len()];
        for (row, out) in x.chunks(cols).zip(output.chunks_mut(cols)) {
            let mut partial = [0f32; 64];
            for (lane, sum) in partial.iter_mut().enumerate() {
                for value in row.iter().skip(lane).step_by(64) {
                    *sum += value;
                }
            }
            let mean = workgroup_sum(&mut partial) / cols as f32;
            let mut partial = [0f32; 64];
            for (lane, sum) in partial.iter_mut().enumerate() {
                for value in row.iter().skip(lane).step_by(64) {
                    let a = value - mean;
                    *sum += a * a;
                }
            }
            let inv = 1. / (workgroup_sum(&mut partial) / cols as f32 + 1e-5).sqrt();
            for (((out, value), w), b) in out.iter_mut().zip(row).zip(weight).zip(bias) {
                *out = (value - mean) * inv * w + b;
            }
        }
        output
    }

    #[test]
    fn transcribe_layer_norm_workgroup_reduction_matches_f64_reference() {
        let source = include_str!("../shaders/transcribe.comp");
        assert!(source.contains("shared float partial[64];"));
        for (rows, cols, center) in [
            (3usize, 1024usize, 0f32),
            (2, 1024, 40.),
            (2, 1000, 3.),
            (2, 50, -2.),
        ] {
            let params = Params {
                op: 101,
                rows: rows as u32,
                cols: cols as u32,
                ..Default::default()
            };
            // One workgroup per row; the shader returns from any extra groups.
            assert_eq!(speech_groups(&params, rows as u32), rows as u32);
            let mut random = Lcg((rows * cols) as u64);
            let x: Vec<_> = random
                .values(rows * cols)
                .into_iter()
                .map(|v| center + 3. * v)
                .collect();
            let weight = random.values(cols);
            let bias = random.values(cols);
            let actual = emulate_layer_norm(&x, &weight, &bias, cols);
            for (row, out) in x.chunks(cols).zip(actual.chunks(cols)) {
                let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / cols as f64;
                let variance = row
                    .iter()
                    .map(|&v| (f64::from(v) - mean).powi(2))
                    .sum::<f64>()
                    / cols as f64;
                let inv = 1. / (variance + 1e-5).sqrt();
                for (k, ((&value, &actual), (&w, &b))) in row
                    .iter()
                    .zip(out)
                    .zip(weight.iter().zip(&bias))
                    .enumerate()
                {
                    let reference = (f64::from(value) - mean) * inv * f64::from(w) + f64::from(b);
                    assert!(
                        (f64::from(actual) - reference).abs() <= 2e-5,
                        "{rows}x{cols} column {k}: {actual} vs {reference}"
                    );
                }
            }
        }
    }

    /// The attention shader sizes its tiles from the lane count alone; every
    /// supported head width must fit its registers and shared arrays.
    #[test]
    fn attention_geometry_fits_shader_arrays_for_every_head_width() {
        for dim in 1..=256 {
            let lanes = attention_lanes(dim);
            assert!([2, 4, 8].contains(&lanes), "{dim}");
            // Each thread holds eight vec4 slices of the head.
            assert!(lanes * 32 >= dim, "{dim}");
            let width = lanes * 8;
            let keys = 512 / width;
            // Key and value tiles hold 512 vec4; keys advance four at a time;
            // partial scores hold 64 per key in 2048 floats.
            assert_eq!(keys * width, 512, "{dim}");
            assert_eq!(keys % 4, 0, "{dim}");
            assert!(keys * 64 <= 2048, "{dim}");
        }
        assert_eq!(attention_lanes(64), 2);
        assert_eq!(attention_lanes(128), 4);
    }
}
