//! Locating the package's own built binaries (worker mocks, probe stubs,
//! feature-off GPU workers) from inside a unit-test executable.

use std::path::PathBuf;

/// The built binary `name` (without the platform executable suffix).
///
/// A unit-test executable sits in `<profile>/deps/`. Cargo puts the package's
/// binaries in `<target>/<profile>/`, which is the deps folder's parent in the
/// default layout. With `CARGO_BUILD_BUILD_DIR` set (the Linux build servers
/// set it), test executables move under `<build-dir>/<profile>/deps/` while
/// binaries stay in `<target>/<profile>/`. So look for `<profile>/<name>`
/// under each ancestor of the deps folder, nearest first, and fall back to the
/// default-layout path so a missing binary is reported where it is expected.
pub(crate) fn built_binary(name: &str) -> PathBuf {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().unwrap().parent().unwrap();
    let profile = profile_dir.file_name().unwrap();
    let nearest = profile_dir.join(&file);
    if nearest.is_file() {
        return nearest;
    }
    profile_dir
        .ancestors()
        .skip(1)
        .map(|ancestor| ancestor.join(profile).join(&file))
        .find(|candidate| candidate.is_file())
        .unwrap_or(nearest)
}
