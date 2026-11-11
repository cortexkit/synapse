//! Dynamically loaded so hosts without NVIDIA drivers need no NVML installation.
use libloading::Library;
use std::ffi::{c_uint, c_void};

type Device = *mut c_void;
type Count = unsafe extern "C" fn(*mut c_uint) -> c_uint;
type Handle = unsafe extern "C" fn(c_uint, *mut Device) -> c_uint;
type Processes = unsafe extern "C" fn(Device, *mut c_uint, *mut Process) -> c_uint;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Process {
    pid: c_uint,
    used_gpu_memory: u64,
    gpu_instance_id: c_uint,
    compute_instance_id: c_uint,
}

pub struct Nvml {
    library: Library,
    count: Count,
    handle: Handle,
    compute: Processes,
    graphics: Processes,
}

impl Nvml {
    pub fn open() -> Option<Self> {
        unsafe {
            let library = Library::new(if cfg!(windows) {
                "nvml.dll"
            } else {
                "libnvidia-ml.so.1"
            })
            .ok()?;
            let init = *library
                .get::<unsafe extern "C" fn() -> c_uint>(b"nvmlInit_v2\0")
                .ok()?;
            let count = *library.get::<Count>(b"nvmlDeviceGetCount_v2\0").ok()?;
            let handle = *library
                .get::<Handle>(b"nvmlDeviceGetHandleByIndex_v2\0")
                .ok()?;
            let compute = *library
                .get::<Processes>(b"nvmlDeviceGetComputeRunningProcesses_v2\0")
                .ok()?;
            let graphics = *library
                .get::<Processes>(b"nvmlDeviceGetGraphicsRunningProcesses_v2\0")
                .ok()?;
            if init() != 0 {
                return None;
            }
            Some(Self {
                library,
                count,
                handle,
                compute,
                graphics,
            })
        }
    }

    pub fn bytes(&self, pid: u32) -> Option<u64> {
        unsafe {
            let mut devices = 0;
            if (self.count)(&mut devices) != 0 {
                return None;
            }
            let mut total: Option<u64> = None;
            for index in 0..devices {
                let mut device = std::ptr::null_mut();
                if (self.handle)(index, &mut device) != 0 {
                    return None;
                }
                let compute = processes(self.compute, device, pid);
                let graphics = processes(self.graphics, device, pid);
                // A process present in both lists has one allocation total on
                // this device, not two independent memory pools.
                if let Some(bytes) = compute.into_iter().chain(graphics).max() {
                    if bytes == u64::MAX {
                        return None;
                    } // NVML_VALUE_NOT_AVAILABLE (WDDM)
                    total = Some(total.unwrap_or(0).checked_add(bytes)?);
                }
            }
            total
        }
    }
}

unsafe fn processes(query: Processes, device: Device, pid: u32) -> Option<u64> {
    let mut count = 0;
    let status = query(device, &mut count, std::ptr::null_mut());
    if count == 0 || (status != 0 && status != 7) {
        return None;
    } // insufficient size
    let mut entries = vec![Process::default(); count as usize + 16];
    count = entries.len() as u32;
    if query(device, &mut count, entries.as_mut_ptr()) != 0 {
        return None;
    }
    entries[..count as usize]
        .iter()
        .find(|p| p.pid == pid)
        .map(|p| p.used_gpu_memory)
}

impl Drop for Nvml {
    fn drop(&mut self) {
        unsafe {
            if let Ok(shutdown) = self
                .library
                .get::<unsafe extern "C" fn() -> c_uint>(b"nvmlShutdown\0")
            {
                shutdown();
            }
        }
    }
}
