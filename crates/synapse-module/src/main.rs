#![forbid(unsafe_code)]

use std::{ffi::OsString, path::PathBuf};

const RESTORE_USAGE: &str = "usage: ck-synapse restore-import <scratch-db> [--into <store-dir>]";

#[tokio::main]
async fn main() {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();

    // Provenance probes must exit before runtime-required arguments are parsed.
    if arguments.iter().any(|argument| argument == "--version") {
        println!(concat!(
            env!("CARGO_BIN_NAME"),
            " ",
            env!("CARGO_PKG_VERSION")
        ));
        return;
    }

    match parse_restore_import(&arguments) {
        Ok(Some((capture, store_directory))) => {
            match synapse_module::restore_import(&capture, store_directory.as_deref()) {
                Ok(report) => println!(
                    "{}",
                    serde_json::to_string(&report).expect("restore report is serializable")
                ),
                Err(error) => {
                    eprintln!("restore import failed: {error}");
                    std::process::exit(1);
                }
            }
        }
        Ok(None) => {
            if let Err(error) = synapse_module::run_from_env().await {
                if matches!(error, synapse_module::ModuleError::SingletonHeld(_)) {
                    std::process::exit(1);
                }
                panic!("synapse module failed: {error}");
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}

fn parse_restore_import(
    arguments: &[OsString],
) -> Result<Option<(PathBuf, Option<PathBuf>)>, String> {
    let Some(command) = arguments.first() else {
        return Ok(None);
    };
    if command != "restore-import" {
        return Ok(None);
    }
    let Some(capture) = arguments.get(1) else {
        return Err(RESTORE_USAGE.to_string());
    };
    if capture.to_string_lossy().starts_with('-') {
        return Err(RESTORE_USAGE.to_string());
    }

    let mut store_directory = None;
    let mut index = 2;
    while index < arguments.len() {
        if arguments[index] != "--into" || store_directory.is_some() {
            return Err(RESTORE_USAGE.to_string());
        }
        let Some(directory) = arguments.get(index + 1) else {
            return Err(RESTORE_USAGE.to_string());
        };
        if directory.to_string_lossy().starts_with('-') {
            return Err(RESTORE_USAGE.to_string());
        }
        store_directory = Some(PathBuf::from(directory));
        index += 2;
    }

    Ok(Some((PathBuf::from(capture), store_directory)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_import_accepts_canonical_and_explicit_target_shapes() {
        assert_eq!(
            parse_restore_import(&["restore-import".into(), "/tmp/capture.db".into()]).unwrap(),
            Some((PathBuf::from("/tmp/capture.db"), None))
        );
        assert_eq!(
            parse_restore_import(&[
                "restore-import".into(),
                "/tmp/capture.db".into(),
                "--into".into(),
                "/tmp/target".into(),
            ])
            .unwrap(),
            Some((
                PathBuf::from("/tmp/capture.db"),
                Some(PathBuf::from("/tmp/target"))
            ))
        );
    }

    #[test]
    fn restore_import_rejects_incomplete_or_extra_arguments() {
        for arguments in [
            vec!["restore-import".into()],
            vec![
                "restore-import".into(),
                "/tmp/capture.db".into(),
                "--into".into(),
            ],
            vec![
                "restore-import".into(),
                "/tmp/capture.db".into(),
                "extra".into(),
            ],
        ] {
            assert_eq!(
                parse_restore_import(&arguments),
                Err(RESTORE_USAGE.to_string())
            );
        }
    }
}
