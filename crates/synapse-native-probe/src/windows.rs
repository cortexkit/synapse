use std::fs::File;
use std::io;
use std::mem::size_of;
use std::os::windows::io::AsRawHandle;

use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
use windows_sys::Win32::Storage::FileSystem::{
    FileBasicInfo, FileIdInfo, GetFileInformationByHandleEx, FILE_BASIC_INFO, FILE_ID_INFO,
};
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

/// Which file an open handle names and when its contents or metadata last
/// changed, as NTFS and ReFS record them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileIdentity {
    /// Serial number of the volume holding the file.
    pub volume_serial: u64,
    /// The file's 128-bit id, unique on its volume.
    pub file_id: [u8; 16],
    /// Last change to the file's data or metadata, in 100 ns units since
    /// 1601. Unlike the last-write time, callers cannot set it back.
    pub change_time: i64,
}

/// Reads the volume serial, file id and change time of an open file. Fails
/// when the file system cannot report them (FAT reports no file id).
pub fn file_identity(file: &File) -> io::Result<FileIdentity> {
    let handle = file.as_raw_handle();
    let mut id = FILE_ID_INFO::default();
    // SAFETY: handle is the open handle of `file`, borrowed for this call;
    // id is an initialized FILE_ID_INFO of the advertised size, written
    // synchronously, and the call keeps no pointer to it.
    let read = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileIdInfo,
            (&raw mut id).cast(),
            size_of::<FILE_ID_INFO>() as u32,
        )
    };
    if read == 0 {
        return Err(io::Error::last_os_error());
    }
    let mut basic = FILE_BASIC_INFO::default();
    // SAFETY: as above, for an initialized FILE_BASIC_INFO.
    let read = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileBasicInfo,
            (&raw mut basic).cast(),
            size_of::<FILE_BASIC_INFO>() as u32,
        )
    };
    if read == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(FileIdentity {
        volume_serial: id.VolumeSerialNumber,
        file_id: id.FileId.Identifier,
        change_time: basic.ChangeTime,
    })
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

    #[test]
    fn file_identity_names_the_file_and_moves_its_change_time_on_write() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "synapse-native-probe-identity-{}",
            std::process::id()
        ));
        let other = path.with_extension("other");
        std::fs::write(&path, b"first").unwrap();
        std::fs::write(&other, b"first").unwrap();
        let before = file_identity(&File::open(&path).unwrap()).unwrap();
        let unrelated = file_identity(&File::open(&other).unwrap()).unwrap();
        // Change-time resolution is coarse on some volumes; wait past it.
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b" second")
            .unwrap();
        let after = file_identity(&File::open(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&other);
        assert_ne!(before.file_id, [0; 16]);
        assert_eq!(before.volume_serial, after.volume_serial);
        assert_eq!(before.file_id, after.file_id);
        assert_ne!(before.file_id, unrelated.file_id);
        assert!(
            after.change_time > before.change_time,
            "{before:?} {after:?}"
        );
    }
}
