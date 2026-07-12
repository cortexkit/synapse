#[cfg(all(target_os = "windows", feature = "vulkan"))]
mod enabled {
    use std::collections::HashMap;
    use std::ffi::{CStr, CString};
    use std::io::Cursor;
    use std::mem::size_of;
    use std::path::PathBuf;
    use std::ptr;
    use std::sync::Arc;
    use std::time::Instant;

    use anyhow::{ensure, Context, Result};
    use ash::{vk, Device, Entry, Instance};

    use super::super::{encode_f16_bits, EncoderLayer, VulkanGemm};

    const DESCRIPTOR_BINDINGS: u32 = 10;
    const PUSH_CONSTANT_BYTES: u32 = 128;

    struct Buffer {
        state: Arc<DeviceState>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        bytes: vk::DeviceSize,
    }

    impl Buffer {
        fn new(state: Arc<DeviceState>, bytes: usize) -> Result<Self> {
            let bytes = bytes.max(4) as vk::DeviceSize;
            let create = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = unsafe { state.device.create_buffer(&create, None)? };
            let requirements = unsafe { state.device.get_buffer_memory_requirements(buffer) };
            let memory_type = state
                .memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .context("no coherent host-visible Vulkan storage memory")?;
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            let memory = unsafe { state.device.allocate_memory(&allocate, None)? };
            unsafe { state.device.bind_buffer_memory(buffer, memory, 0)? };
            Ok(Self {
                state,
                buffer,
                memory,
                bytes,
            })
        }

        fn from_f16(state: Arc<DeviceState>, values: &[f32]) -> Result<Self> {
            let encoded = encode_f16_bits(values);
            let buffer = Self::new(state, encoded.len() * size_of::<u16>())?;
            buffer.write(&encoded)?;
            Ok(buffer)
        }

        fn from_f32(state: Arc<DeviceState>, values: &[f32]) -> Result<Self> {
            let buffer = Self::new(state, std::mem::size_of_val(values))?;
            buffer.write(values)?;
            Ok(buffer)
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
        device: Device,
        queue: vk::Queue,
        queue_family: u32,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        descriptor_layout: vk::DescriptorSetLayout,
        pipeline_layout: vk::PipelineLayout,
        pipeline_cache: vk::PipelineCache,
        pipeline_cache_path: Option<PathBuf>,
        gemm: VulkanGemm,
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
                let queue_family = instance
                    .get_physical_device_queue_family_properties(physical_device)
                    .iter()
                    .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .context("Vulkan GPU has no compute queue")?
                    as u32;

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
                let extension_names = matches!(gemm, VulkanGemm::Cooperative)
                    .then_some(ash::khr::cooperative_matrix::NAME.as_ptr())
                    .into_iter()
                    .collect::<Vec<_>>();
                let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default()
                    .storage_buffer16_bit_access(true);
                let mut float16 =
                    vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
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
                let memory_properties =
                    instance.get_physical_device_memory_properties(physical_device);
                Ok(Arc::new(Self {
                    _entry: entry,
                    instance,
                    device,
                    queue,
                    queue_family,
                    memory_properties,
                    descriptor_layout,
                    pipeline_layout,
                    pipeline_cache,
                    pipeline_cache_path,
                    gemm,
                }))
            }
        }

        fn memory_type(&self, bits: u32, required: vk::MemoryPropertyFlags) -> Option<u32> {
            self.memory_properties.memory_types[..self.memory_properties.memory_type_count as usize]
                .iter()
                .enumerate()
                .find(|(index, memory)| {
                    bits & (1 << index) != 0 && memory.property_flags.contains(required)
                })
                .map(|(index, _)| index as u32)
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

    struct Pipelines {
        plain: vk::Pipeline,
        cooperative: Option<vk::Pipeline>,
        qkv: vk::Pipeline,
        softmax: vk::Pipeline,
        transpose: vk::Pipeline,
        residual_norm: vk::Pipeline,
        gelu: vk::Pipeline,
        pool: vk::Pipeline,
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
            ]
            .into_iter()
            .flatten()
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

    struct Recorder<'a> {
        state: &'a DeviceState,
        command: vk::CommandBuffer,
        descriptor_pool: vk::DescriptorPool,
        descriptor_sets: Vec<vk::DescriptorSet>,
        pipelines: &'a Pipelines,
    }

    impl Recorder<'_> {
        fn dispatch<T: Copy>(
            &mut self,
            pipeline: vk::Pipeline,
            buffers: &[&Buffer],
            params: &T,
            groups: [u32; 3],
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
                self.state
                    .device
                    .cmd_dispatch(self.command, groups[0], groups[1], groups[2]);
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
            };
            // Cooperative B loads are parity-certified for transposed matrices. Attention PV
            // keeps its row-major B operand on the shared plain kernel for every shape policy.
            if matches!(self.state.gemm, VulkanGemm::Cooperative) && transpose_b && k % 16 == 0 {
                self.dispatch(
                    self.pipelines
                        .cooperative
                        .context("cooperative pipeline missing")?,
                    &[a, b, c],
                    &params,
                    groups,
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
                    )?;
                }
                Ok(())
            } else {
                self.dispatch(self.pipelines.plain, &[a, b, c], &params, groups)
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
            };
            let rows = batch * seq;
            let head_dim = hidden / heads;
            let current = &activations.input;
            let next = &activations.x1;
            for layer in layers {
                recorder.gemm(
                    current,
                    &layer.query.weight,
                    &activations.q_raw,
                    rows,
                    hidden,
                    hidden,
                    1,
                    true,
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
            )?;
            unsafe { state.device.end_command_buffer(command)? };
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
            let output = self.activations.pooled.read_f32(self.batch * self.hidden)?;
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
                eprintln!(
                    "Vulkan persistent weights: upload_ms={:.3} layers={} hidden={} intermediate={} storage=f16 norm_params=fp32",
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

    unsafe fn bytes_of<T: Copy>(value: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts((value as *const T).cast(), size_of::<T>()) }
    }
}

#[cfg(not(all(target_os = "windows", feature = "vulkan")))]
mod enabled {
    use std::path::PathBuf;

    use anyhow::{bail, Result};

    use super::super::{EncoderLayer, VulkanGemm};

    pub struct VulkanContext;

    impl VulkanContext {
        pub fn new(_gemm: VulkanGemm, _pipeline_cache_path: Option<PathBuf>) -> Result<Self> {
            bail!("Vulkan provider requires Windows and cargo feature `vulkan`")
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
            bail!("Vulkan provider requires Windows and cargo feature `vulkan`")
        }
    }
}

pub use enabled::VulkanContext;
