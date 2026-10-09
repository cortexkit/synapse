//! Platform-specific data roots shared by the module, model cache, and store.

use std::{ffi::OsString, path::PathBuf};

use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    Linux,
    Windows,
    MacOs,
}

impl Platform {
    #[must_use]
    pub const fn current() -> Self {
        if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::MacOs
        } else {
            Self::Linux
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataRoot {
    Lease,
    ModelCache,
    /// The data home passed to `sqlite_store_path`, which owns the store suffix.
    Store,
}

impl DataRoot {
    fn name(self) -> &'static str {
        match self {
            Self::Lease => "lease",
            Self::ModelCache => "model cache",
            Self::Store => "store",
        }
    }

    fn overrides(self) -> &'static [&'static str] {
        match self {
            Self::Lease => &["CORTEXKIT_LEASE_ROOT"],
            Self::ModelCache => &["CORTEXKIT_MODEL_CACHE", "SYNAPSE_MODEL_CACHE_DIR"],
            // Daemon-supplied storage is handled by the module before this fallback.
            Self::Store => &[],
        }
    }

    fn suffix(self) -> &'static [&'static str] {
        match self {
            Self::Lease => &["cortexkit", "leases"],
            Self::ModelCache => &["cortexkit", "models"],
            Self::Store => &[],
        }
    }
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct DataRootError {
    message: String,
}

/// Resolve a root without consulting the filesystem or the process environment.
///
/// `Store` returns the data home; callers retain the existing `sqlite_store_path`
/// suffix and daemon-storage precedence. Linux and Windows refuse relative roots;
/// macOS retains the historical acceptance of empty and relative overrides.
pub fn resolve_data_root(
    root: DataRoot,
    platform: Platform,
    mut lookup: impl FnMut(&str) -> Option<OsString>,
) -> Result<PathBuf, DataRootError> {
    for variable in root.overrides() {
        if let Some(value) = lookup(variable) {
            if platform == Platform::MacOs {
                // The old overrides used env::var, so non-Unicode values fell through.
                if value.to_str().is_some() {
                    return Ok(PathBuf::from(value));
                }
            } else if !value.is_empty() {
                return validated(root, platform, variable, value);
            }
        }
    }

    let (mut data_home, variable) = match platform {
        Platform::Windows => (
            lookup("LOCALAPPDATA")
                .filter(|value| !value.is_empty())
                .ok_or_else(|| missing(root, "LOCALAPPDATA"))?,
            "LOCALAPPDATA",
        ),
        Platform::Linux | Platform::MacOs => {
            let xdg = if platform == Platform::Linux || root == DataRoot::Store {
                lookup("XDG_DATA_HOME").filter(|value| !value.is_empty())
            } else {
                None
            };
            if let Some(value) = xdg {
                (value, "XDG_DATA_HOME")
            } else {
                let home = lookup("HOME")
                    .filter(|value| {
                        (platform == Platform::MacOs && root != DataRoot::Store)
                            || !value.is_empty()
                    })
                    .ok_or_else(|| {
                        if platform == Platform::MacOs {
                            missing_macos(root)
                        } else {
                            missing(root, "XDG_DATA_HOME and HOME")
                        }
                    })?;
                (append(home, platform, &[".local", "share"]), "HOME")
            }
        }
    };
    if platform != Platform::MacOs {
        validated(root, platform, variable, data_home.clone())?;
    }
    data_home = append(data_home, platform, root.suffix());
    Ok(PathBuf::from(data_home))
}

/// Resolve using this process's native platform and environment.
pub fn resolve_data_root_from_process(root: DataRoot) -> Result<PathBuf, DataRootError> {
    resolve_data_root(root, Platform::current(), |key| std::env::var_os(key))
}

fn missing(root: DataRoot, variables: &str) -> DataRootError {
    DataRootError {
        message: format!("{variables} is unset; cannot resolve {} root", root.name()),
    }
}

fn missing_macos(root: DataRoot) -> DataRootError {
    DataRootError {
        message: match root {
            DataRoot::Lease => "HOME is unset; cannot resolve cortexkit lease root",
            DataRoot::ModelCache => "HOME is unset; cannot resolve model cache",
            DataRoot::Store => "XDG_DATA_HOME and HOME are unset; cannot resolve Synapse store",
        }
        .to_string(),
    }
}

fn validated(
    root: DataRoot,
    platform: Platform,
    variable: &str,
    value: OsString,
) -> Result<PathBuf, DataRootError> {
    // Use the injected platform's syntax, not the host's Path::is_absolute.
    let bytes = value.as_encoded_bytes();
    let absolute = match platform {
        Platform::Windows => windows_absolute(bytes),
        Platform::Linux | Platform::MacOs => bytes.starts_with(b"/"),
    };
    if !absolute {
        return Err(DataRootError {
            message: format!(
                "{} root from {variable} must be absolute: {}",
                root.name(),
                value.to_string_lossy()
            ),
        });
    }
    Ok(PathBuf::from(value))
}

fn windows_absolute(bytes: &[u8]) -> bool {
    let separator = |byte: u8| byte == b'\\' || byte == b'/';
    if bytes.starts_with(b"\\\\?\\")
        && bytes.len() >= 6
        && bytes[4].is_ascii_alphabetic()
        && bytes[5] == b':'
    {
        return bytes.len() >= 7 && separator(bytes[6]);
    }
    if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && separator(bytes[2])
    {
        return true;
    }
    if bytes.len() >= 2 && separator(bytes[0]) && separator(bytes[1]) {
        let mut components = bytes[2..].split(|byte| separator(*byte));
        return components.next().is_some_and(|server| !server.is_empty())
            && components.next().is_some_and(|share| !share.is_empty());
    }
    false
}

fn append(mut base: OsString, platform: Platform, suffix: &[&str]) -> OsString {
    if platform == Platform::MacOs {
        let mut path = PathBuf::from(base);
        for component in suffix {
            path.push(component);
        }
        return path.into_os_string();
    }
    for component in suffix {
        let bytes = base.as_encoded_bytes();
        let ends_in_separator =
            bytes.ends_with(b"/") || (platform == Platform::Windows && bytes.ends_with(b"\\"));
        if !base.is_empty() && !ends_in_separator {
            base.push(if platform == Platform::Windows {
                "\\"
            } else {
                "/"
            });
        }
        base.push(component);
    }
    base
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOTS: [DataRoot; 3] = [DataRoot::Lease, DataRoot::ModelCache, DataRoot::Store];

    fn resolve(
        root: DataRoot,
        platform: Platform,
        environment: &[(&str, &str)],
    ) -> Result<PathBuf, DataRootError> {
        resolve_data_root(root, platform, |key| {
            environment
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| OsString::from(value))
        })
    }

    fn assert_roots(platform: Platform, environment: &[(&str, &str)], expected: [&str; 3]) {
        for (root, expected) in ROOTS.into_iter().zip(expected) {
            assert_eq!(
                resolve(root, platform, environment).unwrap().as_os_str(),
                expected,
                "{platform:?} {root:?} {environment:?}"
            );
        }
    }

    #[test]
    fn linux_roots_follow_xdg_then_home() {
        for environment in [
            vec![("XDG_DATA_HOME", "/native-data")],
            vec![("XDG_DATA_HOME", "/native-data"), ("HOME", "/other-home")],
        ] {
            assert_roots(
                Platform::Linux,
                &environment,
                [
                    "/native-data/cortexkit/leases",
                    "/native-data/cortexkit/models",
                    "/native-data",
                ],
            );
        }
        for environment in [
            vec![("HOME", "/operator")],
            vec![("XDG_DATA_HOME", ""), ("HOME", "/operator")],
        ] {
            assert_roots(
                Platform::Linux,
                &environment,
                [
                    "/operator/.local/share/cortexkit/leases",
                    "/operator/.local/share/cortexkit/models",
                    "/operator/.local/share",
                ],
            );
        }
    }

    #[test]
    fn windows_roots_use_only_localappdata() {
        for environment in [
            vec![("LOCALAPPDATA", r"C:\native-data")],
            vec![
                ("LOCALAPPDATA", r"C:\native-data"),
                ("HOME", r"D:\other-home"),
                ("USERPROFILE", r"E:\profile"),
                ("XDG_DATA_HOME", r"F:\xdg-data"),
            ],
        ] {
            assert_roots(
                Platform::Windows,
                &environment,
                [
                    r"C:\native-data\cortexkit\leases",
                    r"C:\native-data\cortexkit\models",
                    r"C:\native-data",
                ],
            );
        }
    }

    #[test]
    fn missing_or_empty_native_data_home_refuses_and_names_root_and_variables() {
        for platform in [Platform::Linux, Platform::Windows] {
            for environment in [
                vec![],
                vec![("HOME", ""), ("XDG_DATA_HOME", ""), ("LOCALAPPDATA", "")],
                if platform == Platform::Windows {
                    vec![
                        ("HOME", r"C:\home"),
                        ("USERPROFILE", r"C:\profile"),
                        ("XDG_DATA_HOME", r"C:\xdg"),
                    ]
                } else {
                    vec![("LOCALAPPDATA", r"C:\ignored")]
                },
            ] {
                for root in ROOTS {
                    let message = resolve(root, platform, &environment)
                        .unwrap_err()
                        .to_string();
                    assert!(message.contains(root.name()), "{message}");
                    if platform == Platform::Windows {
                        assert!(message.contains("LOCALAPPDATA"), "{message}");
                    } else {
                        assert!(message.contains("XDG_DATA_HOME"), "{message}");
                        assert!(message.contains("HOME"), "{message}");
                    }
                }
            }
        }
    }

    #[test]
    fn overrides_have_precedence_and_empty_overrides_are_unset() {
        for (platform, base, first, second) in [
            (Platform::Linux, "/data", "/first", "/second"),
            (Platform::Windows, r"C:\data", r"D:\first", r"E:\second"),
        ] {
            let native = if platform == Platform::Windows {
                "LOCALAPPDATA"
            } else {
                "XDG_DATA_HOME"
            };
            assert_eq!(
                resolve(
                    DataRoot::Lease,
                    platform,
                    &[(native, base), ("CORTEXKIT_LEASE_ROOT", first)]
                )
                .unwrap()
                .as_os_str(),
                first
            );
            assert_eq!(
                resolve(
                    DataRoot::ModelCache,
                    platform,
                    &[
                        (native, base),
                        ("CORTEXKIT_MODEL_CACHE", first),
                        ("SYNAPSE_MODEL_CACHE_DIR", second)
                    ]
                )
                .unwrap()
                .as_os_str(),
                first
            );
            assert_eq!(
                resolve(
                    DataRoot::ModelCache,
                    platform,
                    &[
                        (native, base),
                        ("CORTEXKIT_MODEL_CACHE", ""),
                        ("SYNAPSE_MODEL_CACHE_DIR", second)
                    ]
                )
                .unwrap()
                .as_os_str(),
                second
            );
            for root in ROOTS {
                let expected = resolve(root, platform, &[(native, base)]).unwrap();
                let actual = resolve(
                    root,
                    platform,
                    &[
                        (native, base),
                        ("CORTEXKIT_LEASE_ROOT", ""),
                        ("CORTEXKIT_MODEL_CACHE", ""),
                        ("SYNAPSE_MODEL_CACHE_DIR", ""),
                    ],
                )
                .unwrap();
                assert_eq!(actual.as_os_str(), expected.as_os_str());
            }
        }
    }

    #[test]
    fn relative_roots_refuse_without_falling_through_and_name_the_source() {
        for platform in [Platform::Linux, Platform::Windows] {
            for (variable, roots) in [
                ("CORTEXKIT_LEASE_ROOT", vec![DataRoot::Lease]),
                ("CORTEXKIT_MODEL_CACHE", vec![DataRoot::ModelCache]),
                ("SYNAPSE_MODEL_CACHE_DIR", vec![DataRoot::ModelCache]),
                (
                    if platform == Platform::Windows {
                        "LOCALAPPDATA"
                    } else {
                        "XDG_DATA_HOME"
                    },
                    ROOTS.to_vec(),
                ),
            ] {
                for root in roots {
                    let message = resolve(
                        root,
                        platform,
                        &[("HOME", "/absolute-home"), (variable, "relative-data")],
                    )
                    .unwrap_err()
                    .to_string();
                    assert!(message.contains(root.name()), "{message}");
                    assert!(message.contains(variable), "{message}");
                }
            }
        }
        for root in ROOTS {
            let message = resolve(root, Platform::Linux, &[("HOME", "relative-home")])
                .unwrap_err()
                .to_string();
            assert!(
                message.contains(root.name()) && message.contains("HOME"),
                "{message}"
            );
        }
        for relative in [
            r"C:drive-relative",
            r"\root-relative",
            "/root-relative",
            r"\\server",
            r"\\?\C:relative",
        ] {
            for root in ROOTS {
                let message = resolve(root, Platform::Windows, &[("LOCALAPPDATA", relative)])
                    .unwrap_err()
                    .to_string();
                assert!(
                    message.contains(root.name()) && message.contains("LOCALAPPDATA"),
                    "{message}"
                );
            }
        }
    }

    #[test]
    fn absolute_roots_are_returned_without_creating_or_requiring_directories() {
        // A unique descendant of a temp directory is absent, even if its ancestor exists.
        let absent =
            std::env::temp_dir().join(format!("synapse-absent-root-{}", std::process::id()));
        assert!(!absent.exists());
        let native = Platform::current();
        if native != Platform::MacOs {
            for root in ROOTS {
                let result = resolve_data_root(root, native, |key| {
                    (key == if native == Platform::Windows {
                        "LOCALAPPDATA"
                    } else {
                        "XDG_DATA_HOME"
                    })
                    .then(|| absent.clone().into_os_string())
                })
                .unwrap();
                assert!(!result.exists());
            }
            for (root, variable) in [
                (DataRoot::Lease, "CORTEXKIT_LEASE_ROOT"),
                (DataRoot::ModelCache, "CORTEXKIT_MODEL_CACHE"),
                (DataRoot::ModelCache, "SYNAPSE_MODEL_CACHE_DIR"),
            ] {
                assert_eq!(
                    resolve_data_root(root, native, |key| {
                        (key == variable).then(|| absent.clone().into_os_string())
                    })
                    .unwrap(),
                    absent
                );
            }
        }
        assert_roots(
            Platform::Windows,
            &[("LOCALAPPDATA", r"\\server\share\data")],
            [
                r"\\server\share\data\cortexkit\leases",
                r"\\server\share\data\cortexkit\models",
                r"\\server\share\data",
            ],
        );
    }

    // Keep the earlier macOS resolvers as reference implementations so the
    // tests can compare env::var versus env::var_os and empty-value behavior.
    fn legacy_macos(
        root: DataRoot,
        mut lookup: impl FnMut(&str) -> Option<OsString>,
    ) -> Result<PathBuf, String> {
        match root {
            DataRoot::Lease => {
                if let Some(value) =
                    lookup("CORTEXKIT_LEASE_ROOT").filter(|value| value.to_str().is_some())
                {
                    return Ok(PathBuf::from(value));
                }
                let home =
                    lookup("HOME").ok_or("HOME is unset; cannot resolve cortexkit lease root")?;
                Ok(PathBuf::from(home)
                    .join(".local")
                    .join("share")
                    .join("cortexkit")
                    .join("leases"))
            }
            DataRoot::ModelCache => {
                for variable in ["CORTEXKIT_MODEL_CACHE", "SYNAPSE_MODEL_CACHE_DIR"] {
                    if let Some(value) = lookup(variable).filter(|value| value.to_str().is_some()) {
                        return Ok(PathBuf::from(value));
                    }
                }
                let home = lookup("HOME").ok_or("HOME is unset; cannot resolve model cache")?;
                Ok(PathBuf::from(home)
                    .join(".local")
                    .join("share")
                    .join("cortexkit")
                    .join("models"))
            }
            DataRoot::Store => lookup("XDG_DATA_HOME")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
                .or_else(|| {
                    lookup("HOME")
                        .filter(|value| !value.is_empty())
                        .map(PathBuf::from)
                        .map(|home| home.join(".local").join("share"))
                })
                .ok_or_else(|| {
                    "XDG_DATA_HOME and HOME are unset; cannot resolve Synapse store".to_string()
                }),
        }
    }

    #[test]
    fn macos_roots_are_byte_identical_including_empty_and_relative_values() {
        let values = [
            None,
            Some(""),
            Some("relative"),
            Some("/absolute"),
            Some("/trailing//"),
        ];
        for home in values {
            for xdg in values {
                for lease in values {
                    for cache in values {
                        for legacy_cache in values {
                            let environment = [
                                ("HOME", home),
                                ("XDG_DATA_HOME", xdg),
                                ("CORTEXKIT_LEASE_ROOT", lease),
                                ("CORTEXKIT_MODEL_CACHE", cache),
                                ("SYNAPSE_MODEL_CACHE_DIR", legacy_cache),
                                ("LOCALAPPDATA", Some("ignored")),
                            ];
                            let lookup = |key: &str| {
                                environment
                                    .iter()
                                    .find(|(name, _)| *name == key)
                                    .and_then(|(_, value)| value.map(OsString::from))
                            };
                            for root in ROOTS {
                                let expected =
                                    legacy_macos(root, lookup).map(|path| path.into_os_string());
                                let actual = resolve_data_root(root, Platform::MacOs, lookup)
                                    .map(|path| path.into_os_string())
                                    .map_err(|error| error.to_string());
                                assert_eq!(actual, expected, "{root:?} {environment:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn macos_non_unicode_overrides_still_fall_through() {
        use std::os::unix::ffi::OsStringExt;

        let lookup = |key: &str| match key {
            "CORTEXKIT_LEASE_ROOT" | "CORTEXKIT_MODEL_CACHE" | "SYNAPSE_MODEL_CACHE_DIR" => {
                Some(OsString::from_vec(vec![b'/', 0xff]))
            }
            "HOME" => Some(OsString::from_vec(vec![b'/', b'h', 0xff])),
            _ => None,
        };
        for root in ROOTS {
            assert_eq!(
                resolve_data_root(root, Platform::MacOs, lookup)
                    .unwrap()
                    .into_os_string(),
                legacy_macos(root, lookup).unwrap().into_os_string()
            );
        }
    }
}
