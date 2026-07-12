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
    struct DeviceReport {
        device_name: String,
        api_version: String,
        driver_version_raw: u32,
        cooperative_matrix: bool,
        cooperative_matrix_robust_buffer_access: bool,
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
