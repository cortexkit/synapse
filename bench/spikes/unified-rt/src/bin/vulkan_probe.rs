#[cfg(feature = "vulkan")]
fn main() -> anyhow::Result<()> {
    use std::ffi::CStr;

    use anyhow::{ensure, Context};
    use ash::{vk, Entry};
    use serde::Serialize;

    #[derive(Serialize)]
    struct MatrixProperty {
        m: u32,
        n: u32,
        k: u32,
        a_type: String,
        b_type: String,
        c_type: String,
        result_type: String,
        saturating_accumulation: bool,
        scope: String,
    }

    #[derive(Serialize)]
    struct MemoryHeap {
        index: u32,
        size_bytes: u64,
        flags: String,
        usage_bytes: Option<u64>,
        budget_bytes: Option<u64>,
    }

    #[derive(Serialize)]
    struct MemoryType {
        index: u32,
        heap_index: u32,
        property_flags: String,
    }

    #[derive(Serialize)]
    struct DeviceReport {
        device_name: String,
        api_version: String,
        driver_version_raw: u32,
        cooperative_matrix: bool,
        cooperative_matrix_robust_buffer_access: bool,
        timestamp_compute_and_graphics: bool,
        timestamp_period_ns: f32,
        compute_queue_timestamp_valid_bits: Option<u32>,
        memory_budget: bool,
        memory_heaps: Vec<MemoryHeap>,
        memory_types: Vec<MemoryType>,
        properties: Vec<MatrixProperty>,
    }

    unsafe {
        let entry = Entry::load().context("load Vulkan loader")?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let create = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = entry
            .create_instance(&create, None)
            .context("create Vulkan instance")?;
        let loader = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
        let mut reports = Vec::new();
        for physical_device in instance.enumerate_physical_devices()? {
            let properties = instance.get_physical_device_properties(physical_device);
            let mut cooperative = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut cooperative);
            instance.get_physical_device_features2(physical_device, &mut features);
            let compute_queue_timestamp_valid_bits = instance
                .get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .find(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|family| family.timestamp_valid_bits);
            let matrix_properties = loader
                .get_physical_device_cooperative_matrix_properties(physical_device)?
                .into_iter()
                .map(|property| MatrixProperty {
                    m: property.m_size,
                    n: property.n_size,
                    k: property.k_size,
                    a_type: format!("{:?}", property.a_type),
                    b_type: format!("{:?}", property.b_type),
                    c_type: format!("{:?}", property.c_type),
                    result_type: format!("{:?}", property.result_type),
                    saturating_accumulation: property.saturating_accumulation != 0,
                    scope: format!("{:?}", property.scope),
                })
                .collect();
            let memory_budget = instance
                .enumerate_device_extension_properties(physical_device)?
                .iter()
                .any(|extension| {
                    CStr::from_ptr(extension.extension_name.as_ptr())
                        == ash::ext::memory_budget::NAME
                });
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let budget = memory_budget.then(|| {
                let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
                let mut properties =
                    vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
                instance.get_physical_device_memory_properties2(physical_device, &mut properties);
                budget
            });
            let memory_heaps = memory_properties.memory_heaps
                [..memory_properties.memory_heap_count as usize]
                .iter()
                .enumerate()
                .map(|(index, heap)| MemoryHeap {
                    index: index as u32,
                    size_bytes: heap.size,
                    flags: format!("{:?}", heap.flags),
                    usage_bytes: budget.as_ref().map(|budget| budget.heap_usage[index]),
                    budget_bytes: budget.as_ref().map(|budget| budget.heap_budget[index]),
                })
                .collect();
            let memory_types = memory_properties.memory_types
                [..memory_properties.memory_type_count as usize]
                .iter()
                .enumerate()
                .map(|(index, memory)| MemoryType {
                    index: index as u32,
                    heap_index: memory.heap_index,
                    property_flags: format!("{:?}", memory.property_flags),
                })
                .collect();
            let device_name = CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();
            reports.push(DeviceReport {
                device_name,
                api_version: format!(
                    "{}.{}.{}",
                    vk::api_version_major(properties.api_version),
                    vk::api_version_minor(properties.api_version),
                    vk::api_version_patch(properties.api_version)
                ),
                driver_version_raw: properties.driver_version,
                cooperative_matrix: cooperative.cooperative_matrix != 0,
                cooperative_matrix_robust_buffer_access: cooperative
                    .cooperative_matrix_robust_buffer_access
                    != 0,
                timestamp_compute_and_graphics: properties.limits.timestamp_compute_and_graphics
                    != 0,
                timestamp_period_ns: properties.limits.timestamp_period,
                compute_queue_timestamp_valid_bits,
                memory_budget,
                memory_heaps,
                memory_types,
                properties: matrix_properties,
            });
        }
        ensure!(!reports.is_empty(), "Vulkan reported no physical devices");
        println!("{}", serde_json::to_string_pretty(&reports)?);
        instance.destroy_instance(None);
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("vulkan-probe requires cargo feature `vulkan`")
}
