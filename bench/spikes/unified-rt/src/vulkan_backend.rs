#[cfg_attr(not(all(target_os = "windows", feature = "vulkan")), allow(dead_code))]
pub struct ModernBertLayer<'a> {
    pub qkv_weight: &'a [f32],
    pub attention_output_weight: &'a [f32],
    pub attention_norm_weight: Option<&'a [f32]>,
    pub mlp_input_weight: &'a [f32],
    pub mlp_output_weight: &'a [f32],
    pub mlp_norm_weight: &'a [f32],
    pub sliding_attention: bool,
}

#[cfg_attr(not(all(target_os = "windows", feature = "vulkan")), allow(dead_code))]
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

    use super::super::{decode_f16_bits, encode_f16_bits, EncoderLayer, VulkanGemm};
    use super::{ModernBertLayer, Qwen3Layer};

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
            self.gemm_offset(a, b, c, m, n, k, batch_count, transpose_b, 0, 0, 0)
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
            };
            let rows = batch * seq;
            let head_dim = hidden / heads;
            for layer in layers {
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
            )?;
            unsafe {
                state.device.end_command_buffer(command)?;
            }
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
            hidden_states.copy_from_slice(&decode_f16_bits(
                &self.activations.x1.read_u16(self.values)?,
            ));
            Ok(())
        }
    }

    impl Drop for ModernShapePlan {
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
                self.layers = layers
                    .iter()
                    .map(|layer| DeviceModernLayer::upload(self.state.clone(), layer, hidden))
                    .collect::<Result<_>>()?;
                self.final_norm = Some(Buffer::from_f32(self.state.clone(), final_norm)?);
                self.model_shape = Some((hidden, intermediate, layers.len()));
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
            };
            let rows = batch * seq;
            let query_width = query_heads * head_dim;
            let kv_width = kv_heads * head_dim;
            let groups = query_heads / kv_heads;
            let group_batches = batch * kv_heads;
            let query_group_values = batch * kv_heads * seq * head_dim;
            let score_group_values = batch * kv_heads * seq * seq;
            for layer in layers {
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
            )?;
            unsafe {
                state.device.end_command_buffer(command)?;
            }
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
            hidden_states.copy_from_slice(&decode_f16_bits(
                &self.activations.x1.read_u16(self.values)?,
            ));
            Ok(())
        }
    }

    impl Drop for QwenShapePlan {
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
                self.layers = layers
                    .iter()
                    .map(|layer| DeviceQwenLayer::upload(self.state.clone(), layer))
                    .collect::<Result<_>>()?;
                self.final_norm = Some(Buffer::from_f32(self.state.clone(), final_norm)?);
                self.model_shape =
                    Some((hidden, query_heads, kv_heads, intermediate, layers.len()));
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

    unsafe fn bytes_of<T: Copy>(value: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts((value as *const T).cast(), size_of::<T>()) }
    }
}

#[cfg(not(all(target_os = "windows", feature = "vulkan")))]
mod enabled {
    use std::path::PathBuf;

    use anyhow::{bail, Result};

    use super::super::{EncoderLayer, VulkanGemm};
    use super::{ModernBertLayer, Qwen3Layer};

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

    pub struct ModernBertContext;

    impl ModernBertContext {
        pub fn new(_gemm: VulkanGemm, _pipeline_cache: Option<PathBuf>) -> Result<Self> {
            bail!("Vulkan provider requires Windows and cargo feature `vulkan`")
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
            bail!("Vulkan provider requires Windows and cargo feature `vulkan`")
        }
    }

    pub struct Qwen3Context;

    impl Qwen3Context {
        pub fn new(_gemm: VulkanGemm, _pipeline_cache: Option<PathBuf>) -> Result<Self> {
            bail!("Vulkan provider requires Windows and cargo feature `vulkan`")
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
            bail!("Vulkan provider requires Windows and cargo feature `vulkan`")
        }
    }
}

pub use enabled::{ModernBertContext, Qwen3Context, VulkanContext};
