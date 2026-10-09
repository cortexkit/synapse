use std::io;
use std::mem::size_of;

use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
use windows_sys::Win32::System::SystemInformation::{
    GetNativeSystemInfo, GlobalMemoryStatusEx, MEMORYSTATUSEX, OSVERSIONINFOW, SYSTEM_INFO,
};

/// The native major, minor and build numbers, without a manifest-dependent version shim.
pub fn version() -> io::Result<(u32, u32, u32)> {
    let mut reading = OSVERSIONINFOW {
        dwOSVersionInfoSize: size_of::<OSVERSIONINFOW>() as u32,
        ..OSVERSIONINFOW::default()
    };
    // SAFETY: reading is an initialized OSVERSIONINFOW of the advertised size,
    // writable for the duration of the call. RtlGetVersion does not retain it.
    let status = unsafe { RtlGetVersion(&mut reading) };
    if status < 0 {
        return Err(io::Error::other(format!(
            "RtlGetVersion returned NTSTATUS {status:#010x}"
        )));
    }
    Ok((
        reading.dwMajorVersion,
        reading.dwMinorVersion,
        reading.dwBuildNumber,
    ))
}

/// The native processor architecture, even when the caller runs under WOW64.
pub fn processor_architecture() -> io::Result<u16> {
    let mut reading = SYSTEM_INFO::default();
    // SAFETY: reading is an initialized, writable SYSTEM_INFO. The call fills
    // the structure synchronously and does not retain the pointer.
    unsafe { GetNativeSystemInfo(&mut reading) };
    // SAFETY: GetNativeSystemInfo writes the processor-architecture member of
    // SYSTEM_INFO's anonymous union; the structure remains live here.
    Ok(unsafe { reading.Anonymous.Anonymous.wProcessorArchitecture })
}

/// Total physical memory in bytes, not available memory or commit capacity.
pub fn total_physical_memory() -> io::Result<u64> {
    let mut reading = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..MEMORYSTATUSEX::default()
    };
    // SAFETY: reading is an initialized MEMORYSTATUSEX with the required length,
    // writable for this synchronous call, which does not retain its pointer.
    if unsafe { GlobalMemoryStatusEx(&mut reading) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(reading.ullTotalPhys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_windows_sources_return_readings() {
        let (major, _, build) = version().expect("RtlGetVersion");
        assert!(major > 0 && build > 0);
        assert!(matches!(
            processor_architecture().expect("GetNativeSystemInfo"),
            9 | 12
        ));
        assert!(total_physical_memory().expect("GlobalMemoryStatusEx") > 0);
    }
}
