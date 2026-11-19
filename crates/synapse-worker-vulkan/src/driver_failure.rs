use ash::vk;
use std::{error::Error, fmt};

#[derive(Debug)]
pub(crate) struct QueueWaitFailed(pub(crate) vk::Result);

impl fmt::Display for QueueWaitFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vulkan queue wait failed: {}", self.0)
    }
}

impl Error for QueueWaitFailed {}

#[derive(Debug)]
pub(crate) struct DeviceAllocationExhausted;

impl fmt::Display for DeviceAllocationExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Vulkan device memory allocation failed: out of device memory")
    }
}

impl Error for DeviceAllocationExhausted {}

pub(crate) fn allocation_failure(error: vk::Result) -> anyhow::Error {
    if error == vk::Result::ERROR_OUT_OF_DEVICE_MEMORY {
        DeviceAllocationExhausted.into()
    } else {
        error.into()
    }
}

pub(crate) fn is_fatal(error: &anyhow::Error) -> bool {
    error.is::<QueueWaitFailed>()
        || error.downcast_ref::<vk::Result>() == Some(&vk::Result::ERROR_DEVICE_LOST)
}
