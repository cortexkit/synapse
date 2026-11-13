//! Locating the package's own built binaries (worker mocks, probe stubs,
//! feature-off GPU workers) from inside a unit-test executable.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// The built binary `name` (without the platform executable suffix).
///
/// Cargo puts the package's binaries in `<target>/<profile>/`, but where the
/// unit-test executable lands depends on the Cargo version and settings:
/// - the classic layout puts it in `<target>/<profile>/deps/`;
/// - Cargo 1.100's build-dir layout puts it in
///   `<build-dir>/<profile>/build/<package>/<hash>/out/`;
/// - with `CARGO_BUILD_BUILD_DIR` set (the Linux build servers set it), that
///   build dir is a separate folder, while binaries stay under `<target>`.
///
/// So find the profile name from the executable's path, then look for
/// `<profile>/<name>` under each ancestor, nearest first. When nothing is
/// found, return the classic-layout path so the error names where it was
/// expected.
pub(crate) fn built_binary(name: &str) -> PathBuf {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    let exe = std::env::current_exe().unwrap();
    let profile_dir = profile_dir(&exe);
    let profile = profile_dir.file_name().unwrap();
    let expected = profile_dir.join(&file);
    if expected.is_file() {
        return expected;
    }
    profile_dir
        .ancestors()
        .skip(1)
        .map(|ancestor| ancestor.join(profile).join(&file))
        .find(|candidate| candidate.is_file())
        .unwrap_or(expected)
}

/// The `<profile>` folder holding this test executable: the parent of a
/// `deps` folder in the classic layout, or of the `build` folder in the
/// build-dir layout. Falls back to the executable's grandparent.
fn profile_dir(exe: &Path) -> PathBuf {
    let dir = exe.parent().unwrap();
    if dir.file_name() == Some(OsStr::new("deps")) {
        return dir.parent().unwrap().to_path_buf();
    }
    exe.ancestors()
        .find(|ancestor| ancestor.file_name() == Some(OsStr::new("build")))
        .and_then(Path::parent)
        .unwrap_or_else(|| dir.parent().unwrap())
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::profile_dir;
    use std::path::Path;

    #[test]
    fn profile_dir_is_found_in_both_cargo_layouts() {
        assert_eq!(
            profile_dir(Path::new("/t/target/debug/deps/synapse_module-0a1b")),
            Path::new("/t/target/debug")
        );
        assert_eq!(
            profile_dir(Path::new(
                "/t/target/ci-test/build/synapse-module/0a1b/out/synapse_module-0a1b"
            )),
            Path::new("/t/target/ci-test")
        );
    }
}
