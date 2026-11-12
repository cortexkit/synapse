//! Dynamically loaded so hosts without NVIDIA drivers need no NVML installation.
use libloading::Library;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_uint, c_void, CStr};

type Device = *mut c_void;
type Count = unsafe extern "C" fn(*mut c_uint) -> c_uint;
type Handle = unsafe extern "C" fn(c_uint, *mut Device) -> c_uint;
type Processes = unsafe extern "C" fn(Device, *mut c_uint, *mut Process) -> c_uint;
type GetMemory = unsafe extern "C" fn(Device, *mut DeviceMemory) -> c_uint;
type GetUuid = unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_uint;

#[repr(C)]
#[derive(Default)]
struct DeviceMemory {
    total: u64,
    free: u64,
    used: u64,
}

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
    memory: GetMemory,
    uuid: GetUuid,
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
            let memory = *library
                .get::<GetMemory>(b"nvmlDeviceGetMemoryInfo\0")
                .ok()?;
            let uuid = *library.get::<GetUuid>(b"nvmlDeviceGetUUID\0").ok()?;
            if init() != 0 {
                return None;
            }
            Some(Self {
                library,
                count,
                handle,
                compute,
                graphics,
                memory,
                uuid,
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

    pub fn device_bytes(&self) -> Option<BTreeMap<String, u64>> {
        unsafe {
            let mut count = 0;
            if (self.count)(&mut count) != 0 || count == 0 {
                return None;
            }
            let mut devices = BTreeMap::new();
            for index in 0..count {
                let mut device = std::ptr::null_mut();
                if (self.handle)(index, &mut device) != 0 {
                    return None;
                }
                let mut memory = DeviceMemory::default();
                let mut uuid = [0 as c_char; 96];
                if (self.memory)(device, &mut memory) != 0
                    || (self.uuid)(device, uuid.as_mut_ptr(), uuid.len() as u32) != 0
                    || memory.used > memory.total
                    || memory.used == u64::MAX
                    || uuid.last() != Some(&0)
                {
                    return None;
                }
                let id = CStr::from_ptr(uuid.as_ptr()).to_str().ok()?.to_owned();
                if id.is_empty() || devices.insert(id, memory.used).is_some() {
                    return None;
                }
            }
            Some(devices)
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
