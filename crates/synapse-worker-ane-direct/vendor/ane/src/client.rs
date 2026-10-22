use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::os::unix::io::AsRawFd;
use std::path::{Component, Path, PathBuf};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSData, NSDictionary, NSNumber, NSQualityOfService, NSString};
use objc2_io_surface::IOSurface;

use crate::Error;
use crate::ane_in_memory_model::ANEInMemoryModel;
use crate::ane_in_memory_model_descriptor::ANEInMemoryModelDescriptor;
use crate::executable::Executable;
use crate::graph::Graph;
use crate::io_surface::IOSurfaceExt;

static FRAMEWORK_INIT: Once = Once::new();
static FRAMEWORK_OK: AtomicBool = AtomicBool::new(false);

fn ensure_framework() -> Result<(), Error> {
    FRAMEWORK_INIT.call_once(|| {
        let Ok(path) = CString::new(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        ) else {
            return;
        };
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW) };
        if !handle.is_null() {
            FRAMEWORK_OK.store(true, Ordering::Release);
        }
    });
    if FRAMEWORK_OK.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err(Error::FrameworkLoad)
    }
}

pub fn nsdata_on_surface(data: &[u8]) -> (Retained<NSData>, Retained<IOSurface>) {
    let surface = IOSurface::with_byte_count(data.len());
    surface.write_bytes(data);
    let nsdata = unsafe {
        NSData::dataWithBytesNoCopy_length_freeWhenDone(surface.baseAddress(), data.len(), false)
    };
    (nsdata, surface)
}

pub fn compile_network(
    graph: &Graph,
    quality_of_service: NSQualityOfService,
) -> Result<Executable, Error> {
    ensure_framework()?;

    let (ops, shapes) = graph.to_ops_and_shapes();
    let (mil_text, weight_bytes) = crate::ops::mil::emit_mil(&ops, &shapes);

    let (mil_data, _mil_surface) = nsdata_on_surface(mil_text.as_bytes());

    let _weight_surface: Option<Retained<IOSurface>>;
    let weights_dict: Retained<NSDictionary<NSString, AnyObject>> = if weight_bytes.is_empty() {
        _weight_surface = None;
        NSDictionary::new()
    } else {
        let (weight_data, weight_surface) = nsdata_on_surface(&weight_bytes);
        _weight_surface = Some(weight_surface);
        let offset = NSNumber::new_u64(0);
        let entry: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
            &[&*NSString::from_str("offset"), &*NSString::from_str("data")],
            &[
                offset.as_ref() as &AnyObject,
                weight_data.as_ref() as &AnyObject,
            ],
        );
        let key = NSString::from_str("@model_path/weights/weight.bin");
        NSDictionary::from_slices(&[&*key], &[entry.as_ref() as &AnyObject])
    };

    let descriptor = ANEInMemoryModelDescriptor::new(&mil_data, Some(&weights_dict))
        .ok_or(Error::ModelCreation)?;

    let model = ANEInMemoryModel::with_descriptor(&descriptor).ok_or(Error::ModelCreation)?;

    let identifier = model.hex_string_identifier().map(|id| id.to_string());
    with_compile_dir(
        &std::env::temp_dir(),
        identifier.as_deref(),
        mil_text.as_bytes(),
        &weight_bytes,
        || {
            model
                .compile(quality_of_service)
                .map_err(|error| Error::Compile(error.localizedDescription().to_string()))?;
            model
                .load(quality_of_service)
                .map_err(|error| Error::Load(error.localizedDescription().to_string()))
        },
    )?;

    Ok(Executable {
        inner: model,
        qos: quality_of_service,
        cached_request: std::cell::UnsafeCell::new(None),
        stats_request: std::cell::UnsafeCell::new(None),
        perf_stats: std::cell::UnsafeCell::new(None),
    })
}

/// Gives the private compiler the source files of one layer program, runs
/// `compile_and_load`, then removes the files again.
///
/// The compiler reads the program's MIL text and weights from
/// `<temp_root>/<identifier>/` (`model.mil` and `weights/weight.bin`) and
/// writes its own outputs (`net.plist` and a `data` file as large as the
/// weights) beside them. A loaded program does not read that directory again:
/// on an M4 with macOS 27.0 (26A428), outputs stayed byte-identical after the
/// directories were deleted, including across eviction, recompilation and
/// unload. So the directory is removed right after the load, and also when
/// writing the files, the compile or the load fails. Left in place, it stays
/// on disk at least as long as the program stays loaded (about 1.7 GB per
/// resident Qwen3 shape), and processes that ended without unloading their
/// programs left such directories behind for good.
///
/// Only this exact path is removed, never anything found by name or pattern:
/// other applications use the same private compiler and the same per-user
/// temp folder.
///
/// On that OS build every compile got a fresh identifier, even for an
/// identical program, so two compiles normally never share a directory.
/// Nothing guarantees that, so each compile holds a shared `flock` on the
/// directory while it uses it. If two compiles ever share one, whoever
/// finishes last can take the lock exclusively and removes it; a compile that
/// finds it still shared leaves it to the other.
pub(crate) fn with_compile_dir<T>(
    temp_root: &Path,
    identifier: Option<&str>,
    mil_text: &[u8],
    weight_bytes: &[u8],
    compile_and_load: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    // Without an identifier there is no directory to prepare, as upstream.
    let Some(identifier) = identifier else {
        return compile_and_load();
    };
    let dir = CompileDir::acquire(temp_root, identifier)?;
    dir.write_sources(mil_text, weight_bytes)?;
    // `dir` is dropped after this returns; dropping it removes the directory
    // unless another compile of the same program still holds it.
    compile_and_load()
}

/// One compile's hold on its program directory; dropping it removes the
/// directory unless another compile of the same program still holds it.
struct CompileDir {
    path: PathBuf,
    handle: File,
}

impl CompileDir {
    /// How often to start over when a finishing compile of the same program
    /// removes the directory between our create and our lock.
    const ATTEMPTS: usize = 16;

    fn acquire(temp_root: &Path, identifier: &str) -> Result<Self, Error> {
        // The identifier becomes a path that is later removed recursively, so
        // it must name exactly one entry inside `temp_root`.
        let mut components = Path::new(identifier).components();
        if !matches!(
            (components.next(), components.next()),
            (Some(Component::Normal(_)), None)
        ) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("compiler identifier {identifier:?} is not a single file name"),
            )
            .into());
        }
        let path = temp_root.join(identifier);
        for _ in 0..Self::ATTEMPTS {
            std::fs::create_dir_all(&path)?;
            let handle = match File::open(&path) {
                Ok(handle) => handle,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            flock(&handle, libc::LOCK_SH)?;
            // If the path no longer names the directory we locked, a compile
            // that finished meanwhile removed it; start over with a new one.
            if names_same_directory(&path, &handle)? {
                return Ok(Self { path, handle });
            }
        }
        Err(io::Error::other(format!(
            "{} kept being removed while it was being prepared",
            path.display()
        ))
        .into())
    }

    fn write_sources(&self, mil_text: &[u8], weight_bytes: &[u8]) -> io::Result<()> {
        std::fs::write(self.path.join("model.mil"), mil_text)?;
        if !weight_bytes.is_empty() {
            let weights_dir = self.path.join("weights");
            std::fs::create_dir_all(&weights_dir)?;
            std::fs::write(weights_dir.join("weight.bin"), weight_bytes)?;
        }
        Ok(())
    }
}

impl Drop for CompileDir {
    fn drop(&mut self) {
        // Release the shared lock before asking for the exclusive one, so that
        // of two compiles finishing together at least one gets it: a failed
        // attempt leaves this one holding nothing.
        let _ = flock(&self.handle, libc::LOCK_UN);
        match flock(&self.handle, libc::LOCK_EX | libc::LOCK_NB) {
            Ok(()) => {}
            // Another compile of the same program still uses the directory
            // and removes it when it finishes.
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return,
            Err(error) => {
                eprintln!(
                    "ane: kept compiler temp directory {}: lock failed: {error}",
                    self.path.display()
                );
                return;
            }
        }
        // Holding the exclusive lock, nobody else can remove or replace the
        // directory. If the path is already gone, or names a newer directory,
        // there is nothing of ours to remove.
        if let Ok(true) = names_same_directory(&self.path, &self.handle)
            && let Err(error) = std::fs::remove_dir_all(&self.path)
        {
            eprintln!(
                "ane: could not remove compiler temp directory {}: {error}",
                self.path.display()
            );
        }
    }
}

fn flock(handle: &File, operation: libc::c_int) -> io::Result<()> {
    loop {
        if unsafe { libc::flock(handle.as_raw_fd(), operation) } == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

/// Whether `path` still names the directory `handle` was opened on.
fn names_same_directory(path: &Path, handle: &File) -> io::Result<bool> {
    let current = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    let held = handle.metadata()?;
    Ok(current.dev() == held.dev() && current.ino() == held.ino())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty stand-in for the per-user temp folder, removed on drop.
    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "ane-compile-dir-test-{}-{name}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const ID: &str = "AB12_CD34_EF56";

    /// Stands in for the private compile and load: checks that the source
    /// files are in place, then writes the two files the real compiler adds.
    fn fake_compiler_outputs(dir: &Path) {
        assert_eq!(std::fs::read(dir.join("model.mil")).unwrap(), b"program");
        assert_eq!(
            std::fs::read(dir.join("weights/weight.bin")).unwrap(),
            b"weights"
        );
        std::fs::write(dir.join("net.plist"), b"plist").unwrap();
        std::fs::write(dir.join("data"), b"weights").unwrap();
    }

    #[test]
    fn compile_dir_is_removed_after_a_successful_load() {
        let root = TempRoot::new("success");
        // Somebody else's compiler directory in the same temp folder.
        let other = root.0.join("0000_1111_2222");
        std::fs::create_dir_all(other.join("weights")).unwrap();
        std::fs::write(other.join("model.mil"), b"theirs").unwrap();

        let dir = root.0.join(ID);
        let result = with_compile_dir(&root.0, Some(ID), b"program", b"weights", || {
            fake_compiler_outputs(&dir);
            Ok(7)
        });

        assert_eq!(result.unwrap(), 7);
        assert!(!dir.exists(), "the program's directory must be removed");
        assert_eq!(std::fs::read(other.join("model.mil")).unwrap(), b"theirs");
    }

    #[test]
    fn compile_dir_is_removed_after_a_failed_compile_or_load() {
        let root = TempRoot::new("failure");
        let dir = root.0.join(ID);
        for failure in [
            Error::Compile("rejected".into()),
            Error::Load("refused".into()),
        ] {
            let expected = failure.to_string();
            let result: Result<(), Error> =
                with_compile_dir(&root.0, Some(ID), b"program", b"weights", || {
                    fake_compiler_outputs(&dir);
                    Err(failure)
                });
            assert_eq!(result.unwrap_err().to_string(), expected);
            assert!(!dir.exists(), "{expected}: the directory must be removed");
        }
    }

    #[test]
    fn compile_dir_is_removed_when_writing_the_sources_fails() {
        let root = TempRoot::new("write-failure");
        let dir = root.0.join(ID);
        std::fs::create_dir_all(&dir).unwrap();
        // A file where the weights directory belongs makes the write fail.
        std::fs::write(dir.join("weights"), b"in the way").unwrap();
        let mut called = false;
        let result = with_compile_dir(&root.0, Some(ID), b"program", b"weights", || {
            called = true;
            Ok(())
        });
        assert!(matches!(result, Err(Error::Io(_))), "{result:?}");
        assert!(!called, "nothing is compiled from incomplete sources");
        assert!(!dir.exists(), "the directory must be removed");
    }

    #[test]
    fn compile_dir_stays_while_another_compile_of_the_same_program_uses_it() {
        let root = TempRoot::new("shared");
        let dir = root.0.join(ID);
        // A second process compiling the same program holds the directory.
        let other = CompileDir::acquire(&root.0, ID).unwrap();
        other.write_sources(b"program", b"weights").unwrap();

        with_compile_dir(&root.0, Some(ID), b"program", b"weights", || {
            fake_compiler_outputs(&dir);
            Ok(())
        })
        .unwrap();
        assert!(
            dir.join("weights/weight.bin").exists(),
            "the other compile still reads these files"
        );

        drop(other);
        assert!(!dir.exists(), "the last compile to finish removes it");
    }

    #[test]
    fn no_identifier_means_no_directory() {
        let root = TempRoot::new("no-identifier");
        let mut called = false;
        with_compile_dir(&root.0, None, b"program", b"weights", || {
            called = true;
            Ok(())
        })
        .unwrap();
        assert!(called);
        assert_eq!(std::fs::read_dir(&root.0).unwrap().count(), 0);
    }

    #[test]
    fn identifier_must_be_a_single_file_name() {
        let root = TempRoot::new("bad-identifier");
        let inner = root.0.join("inner");
        std::fs::create_dir_all(&inner).unwrap();
        for identifier in ["..", ".", "", "a/b", "/abs"] {
            let result = with_compile_dir(&inner, Some(identifier), b"p", b"w", || Ok(()));
            assert!(
                matches!(&result, Err(Error::Io(error)) if error.kind() == io::ErrorKind::InvalidInput),
                "{identifier:?}: {result:?}"
            );
        }
        assert!(inner.exists() && root.0.exists());
    }
}
