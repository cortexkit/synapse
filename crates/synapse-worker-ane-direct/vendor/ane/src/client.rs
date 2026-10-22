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

/// How a layer program got onto the Neural Engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramOrigin {
    /// aned already held the compiled program for this identifier, and it
    /// was loaded without writing sources or compiling.
    CacheHit,
    /// The program was compiled from its sources, then loaded.
    Compiled,
}

impl ProgramOrigin {
    /// The name used in logs and profile records: `cache_hit` or `compiled`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CacheHit => "cache_hit",
            Self::Compiled => "compiled",
        }
    }
}

pub fn compile_network(
    graph: &Graph,
    quality_of_service: NSQualityOfService,
) -> Result<Executable, Error> {
    compile_network_gated(graph, quality_of_service, || Ok::<(), Error>(()))
}

/// [`compile_network`], calling `before_compile` just before it would write
/// the sources and compile. If `before_compile` fails, its error is returned
/// unchanged and nothing is written or compiled. A program that aned already
/// holds is loaded without calling it, because that load writes nothing.
pub fn compile_network_gated<E: From<Error>>(
    graph: &Graph,
    quality_of_service: NSQualityOfService,
    before_compile: impl FnOnce() -> Result<(), E>,
) -> Result<Executable, E> {
    ensure_framework()?;
    let sources = ProgramSources::new(graph)?;
    let new_program = || {
        ANEInMemoryModel::with_descriptor(&sources.descriptor)
            .map(|model| InMemoryProgram {
                model,
                qos: quality_of_service,
            })
            .ok_or(Error::ModelCreation)
    };
    let (program, origin) = load_program(
        new_program,
        &std::env::temp_dir(),
        sources.mil_text.as_bytes(),
        &sources.weight_bytes,
        before_compile,
    )?;

    Ok(Executable {
        inner: program.model,
        qos: quality_of_service,
        origin,
        cached_request: std::cell::UnsafeCell::new(None),
        stats_request: std::cell::UnsafeCell::new(None),
        perf_stats: std::cell::UnsafeCell::new(None),
    })
}

/// The identifier aned files this graph's compiled program under
/// (`<MIL hash>_<weights hash>_<options hash>`), computed from the same
/// descriptor [`compile_network`] builds. Nothing is compiled or loaded and
/// aned is not asked about its cache.
#[doc(hidden)]
pub fn program_identifier(graph: &Graph) -> Result<Option<String>, Error> {
    ensure_framework()?;
    let sources = ProgramSources::new(graph)?;
    let model =
        ANEInMemoryModel::with_descriptor(&sources.descriptor).ok_or(Error::ModelCreation)?;
    Ok(model.hex_string_identifier().map(|id| id.to_string()))
}

/// One layer program's MIL text and packed weights, and the in-memory model
/// descriptor built from them.
struct ProgramSources {
    mil_text: String,
    weight_bytes: Box<[u8]>,
    descriptor: Retained<ANEInMemoryModelDescriptor>,
    // The descriptor's NSData objects point into these surfaces without
    // copying, so the surfaces are kept as long as the sources are.
    _mil_surface: Retained<IOSurface>,
    _weight_surface: Option<Retained<IOSurface>>,
}

impl ProgramSources {
    fn new(graph: &Graph) -> Result<Self, Error> {
        let (ops, shapes) = graph.to_ops_and_shapes();
        let (mil_text, weight_bytes) = crate::ops::mil::emit_mil(&ops, &shapes);

        let (mil_data, mil_surface) = nsdata_on_surface(mil_text.as_bytes());

        let weight_surface: Option<Retained<IOSurface>>;
        let weights_dict: Retained<NSDictionary<NSString, AnyObject>> = if weight_bytes.is_empty() {
            weight_surface = None;
            NSDictionary::new()
        } else {
            let (weight_data, surface) = nsdata_on_surface(&weight_bytes);
            weight_surface = Some(surface);
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
        Ok(Self {
            mil_text,
            weight_bytes,
            descriptor,
            _mil_surface: mil_surface,
            _weight_surface: weight_surface,
        })
    }
}

/// The private calls that getting one in-memory program onto the Neural
/// Engine needs. [`load_program`] decides between them; the unit tests drive
/// it with a stand-in, because the real calls need aned and the hardware.
pub(crate) trait LayerProgram {
    /// aned's identifier for the program, which also names its compiler
    /// temp directory.
    fn identifier(&self) -> Option<String>;
    /// Whether aned already holds the compiled program.
    fn compiled_model_exists(&self) -> bool;
    fn compile(&self) -> Result<(), Error>;
    fn load(&self) -> Result<(), Error>;
}

struct InMemoryProgram {
    model: Retained<ANEInMemoryModel>,
    qos: NSQualityOfService,
}

impl LayerProgram for InMemoryProgram {
    fn identifier(&self) -> Option<String> {
        self.model.hex_string_identifier().map(|id| id.to_string())
    }

    fn compiled_model_exists(&self) -> bool {
        self.model.compiled_model_exists()
    }

    fn compile(&self) -> Result<(), Error> {
        self.model
            .compile(self.qos)
            .map_err(|error| Error::Compile(error.localizedDescription().to_string()))
    }

    fn load(&self) -> Result<(), Error> {
        self.model
            .load(self.qos)
            .map_err(|error| Error::Load(error.localizedDescription().to_string()))
    }
}

/// Gets one program onto the Neural Engine, reusing aned's compiled copy when
/// there is one.
///
/// aned keeps compiled in-memory programs on disk, keyed by the program's
/// identifier (a hash of its MIL text, weights and options) and the calling
/// binary's code-signing identifier. When it holds this program, `load`
/// alone uses that copy: nothing is written and nothing is compiled. On
/// macOS 27.0 (26A425), such a load took 3.5–45 ms against 0.37–15.8 s for a
/// compile, with byte-identical outputs.
///
/// Otherwise this compiles as before: write the sources, compile, load,
/// remove the sources ([`with_compile_dir`]). Two shortcuts are wrong and are
/// not taken. A `load` without a cached copy makes the framework write its
/// own sources, which aned then rejects (`verifyBundleAtPath` hash
/// mismatch). And `compile` never consults the cache, so compiling a cached
/// program repeats the whole compile.
///
/// If the cached load fails because the copy is unusable (for example aned
/// evicted it after the check, or it fails aned's bundle verification), the
/// program is compiled from a fresh model object, since the failed load may
/// have left the first one in an unknown state. A resource refusal is
/// different: compiling would not make room, so every layer would compile and
/// then be refused the same way while holding the machine's compiler. That
/// error is returned at once and unchanged, for the caller to evict and retry.
/// `before_compile` runs only on the compile path, right before the sources
/// are written.
pub(crate) fn load_program<P: LayerProgram, E: From<Error>>(
    mut new_program: impl FnMut() -> Result<P, Error>,
    temp_root: &Path,
    mil_text: &[u8],
    weight_bytes: &[u8],
    before_compile: impl FnOnce() -> Result<(), E>,
) -> Result<(P, ProgramOrigin), E> {
    let mut program = new_program()?;
    if program.compiled_model_exists() {
        match program.load() {
            Ok(()) => return Ok((program, ProgramOrigin::CacheHit)),
            Err(error) if error.is_resource_refusal() => return Err(error.into()),
            Err(error) => {
                eprintln!(
                    "ane: loading cached program {} failed, compiling it instead: {error}",
                    program.identifier().as_deref().unwrap_or("<no identifier>")
                );
                program = new_program()?;
            }
        }
    }
    before_compile()?;
    let identifier = program.identifier();
    with_compile_dir(
        temp_root,
        identifier.as_deref(),
        mil_text,
        weight_bytes,
        || {
            program.compile()?;
            program.load()
        },
    )?;
    Ok((program, ProgramOrigin::Compiled))
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
/// The identifier is a hash of the program's bytes, and the graph emits
/// identical bytes for an identical program, so two processes compiling the
/// same layer at the same time share one directory. Each compile therefore
/// holds a shared `flock` on the directory while it uses it. If two compiles
/// share one, whoever
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

    /// Stand-in for one `_ANEInMemoryModel`: answers the cache query as told
    /// and records every call in a log shared with the test.
    struct FakeProgram {
        cached: bool,
        /// The description a cached load fails with, if it fails.
        cached_load_error: Option<&'static str>,
        dir: PathBuf,
        log: std::rc::Rc<std::cell::RefCell<Vec<&'static str>>>,
    }

    impl LayerProgram for FakeProgram {
        fn identifier(&self) -> Option<String> {
            Some(ID.into())
        }

        fn compiled_model_exists(&self) -> bool {
            self.log.borrow_mut().push("exists");
            self.cached
        }

        fn compile(&self) -> Result<(), Error> {
            self.log.borrow_mut().push("compile");
            fake_compiler_outputs(&self.dir);
            Ok(())
        }

        fn load(&self) -> Result<(), Error> {
            if self.cached
                && let Some(message) = self.cached_load_error
            {
                self.log.borrow_mut().push("load-failed");
                return Err(Error::Load(message.into()));
            }
            self.log.borrow_mut().push("load");
            Ok(())
        }
    }

    /// How a cached load fails when aned's copy cannot be used.
    const UNUSABLE_COPY: &str = "verifyBundleAtPath: hash mismatch";
    /// How a load fails when the Neural Engine has no room for the program.
    const NO_RESOURCES: &str = "no ANE resources (transient; retry)";

    #[derive(Debug)]
    enum GateError {
        Refused(&'static str),
        Ane(Error),
    }

    impl From<Error> for GateError {
        fn from(error: Error) -> Self {
            Self::Ane(error)
        }
    }

    /// Runs `load_program` with fake programs: the first one answers the
    /// cache query with `cached`; any later one (the compile after a failed
    /// cached load) is a miss. Returns the outcome and the call log.
    fn run_load_program(
        root: &TempRoot,
        cached: bool,
        cached_load_error: Option<&'static str>,
        gate: Result<(), GateError>,
    ) -> (Result<ProgramOrigin, GateError>, Vec<&'static str>) {
        let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut created = 0;
        let new_program = || {
            log.borrow_mut().push("new");
            created += 1;
            Ok(FakeProgram {
                cached: cached && created == 1,
                cached_load_error,
                dir: root.0.join(ID),
                log: log.clone(),
            })
        };
        let result = load_program(new_program, &root.0, b"program", b"weights", || {
            log.borrow_mut().push("gate");
            gate
        })
        .map(|(_, origin)| origin);
        let calls = log.borrow().clone();
        (result, calls)
    }

    #[test]
    fn a_cached_program_is_loaded_without_writing_or_compiling() {
        let root = TempRoot::new("cache-hit");
        let (result, calls) = run_load_program(&root, true, None, Ok(()));
        assert_eq!(result.unwrap(), ProgramOrigin::CacheHit);
        // No temp-space gate, no sources, no compile.
        assert_eq!(calls, ["new", "exists", "load"]);
        assert_eq!(std::fs::read_dir(&root.0).unwrap().count(), 0);
    }

    #[test]
    fn an_uncached_program_is_compiled_then_loaded() {
        let root = TempRoot::new("cache-miss");
        let (result, calls) = run_load_program(&root, false, None, Ok(()));
        assert_eq!(result.unwrap(), ProgramOrigin::Compiled);
        // No load is tried before the compile: on a miss it would make the
        // framework write sources that aned rejects.
        assert_eq!(calls, ["new", "exists", "gate", "compile", "load"]);
        assert!(!root.0.join(ID).exists(), "the sources must be removed");
    }

    #[test]
    fn a_failed_cached_load_falls_back_to_compiling_a_fresh_model() {
        let root = TempRoot::new("cache-hit-fails");
        let (result, calls) = run_load_program(&root, true, Some(UNUSABLE_COPY), Ok(()));
        assert_eq!(result.unwrap(), ProgramOrigin::Compiled);
        assert_eq!(
            calls,
            [
                "new",
                "exists",
                "load-failed",
                "new",
                "gate",
                "compile",
                "load"
            ]
        );
        assert!(!root.0.join(ID).exists(), "the sources must be removed");
    }

    #[test]
    fn a_resource_refusal_on_a_cached_load_is_returned_without_compiling() {
        // Compiling cannot make room on the Neural Engine: the caller must
        // evict and retry, so the refusal comes back at once and unchanged.
        let root = TempRoot::new("cache-hit-no-resources");
        let (result, calls) = run_load_program(&root, true, Some(NO_RESOURCES), Ok(()));
        assert!(
            matches!(&result, Err(GateError::Ane(Error::Load(message))) if message == NO_RESOURCES),
            "{result:?}"
        );
        assert_eq!(calls, ["new", "exists", "load-failed"]);
        assert_eq!(std::fs::read_dir(&root.0).unwrap().count(), 0);
    }

    #[test]
    fn only_a_load_without_ane_resources_is_a_resource_refusal() {
        assert!(Error::Load(NO_RESOURCES.into()).is_resource_refusal());
        assert!(!Error::Load(UNUSABLE_COPY.into()).is_resource_refusal());
        assert!(!Error::Compile(NO_RESOURCES.into()).is_resource_refusal());
        assert!(!Error::Evaluate(NO_RESOURCES.into()).is_resource_refusal());
    }

    #[test]
    fn a_refused_compile_writes_and_compiles_nothing() {
        let root = TempRoot::new("gate-refuses");
        for (cached, cached_load_error) in [(false, None), (true, Some(UNUSABLE_COPY))] {
            let (result, calls) = run_load_program(
                &root,
                cached,
                cached_load_error,
                Err(GateError::Refused("no room")),
            );
            assert!(
                matches!(result, Err(GateError::Refused("no room"))),
                "{result:?}"
            );
            assert_eq!(calls.last(), Some(&"gate"), "{calls:?}");
            assert!(!calls.contains(&"compile"), "{calls:?}");
            assert_eq!(std::fs::read_dir(&root.0).unwrap().count(), 0);
        }
    }
}
