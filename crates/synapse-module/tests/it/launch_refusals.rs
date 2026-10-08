//! Startup refusals, asserted on the real binary.
//!
//! The module resolves its own id with a documented fallback to the default for
//! unsupervised local runs, and refuses only when a launch nonce says the daemon
//! spawned it. The fleet logger resolves an id too, from the same variable, but
//! with no fallback: `Config::from_env` refuses outright when `SUBC_MODULE_ID` is
//! absent. Wiring the logger straight to that refusal turns a supported local run
//! into a startup panic, and nothing in production notices because the daemon
//! always injects the variable. These tests pin the two arms apart.

use super::common;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Runs the module binary in an isolated home so it can never touch the
/// operator's live logs, lease or store, and returns its combined output.
fn launch(home: &Path, vars: &[(&str, &str)]) -> String {
    launch_command(home, vars, |_| {})
}

/// `launch`, with a hook that can adjust the command last, after every other
/// setting, which is where the launch-nonce handoff must be installed.
fn launch_command(home: &Path, vars: &[(&str, &str)], finish: impl FnOnce(&mut Command)) -> String {
    let mut command = Command::new(
        synapse_core::dev_binary::ckdev_binary(env!("CARGO_BIN_EXE_ck-synapse"), home).unwrap(),
    );
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", home)
        .env("XDG_DATA_HOME", home.join("data"))
        .env("XDG_CONFIG_HOME", home.join("config"));
    for (key, value) in vars {
        command.env(key, value);
    }
    finish(&mut command);
    let output = command.output().expect("module binary runs");
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Uses the shared helper so the directory lands under the one test root the
/// harness sweeps by age, rather than loose in the temp dir.
fn isolated_home(label: &str) -> PathBuf {
    let home = common::unique_temp_dir(&format!("launch-{label}"));
    std::fs::create_dir_all(&home).expect("create isolated home");
    home
}

#[test]
fn unsupervised_launch_gets_past_the_logger() {
    let home = isolated_home("unsupervised");
    let output = launch(&home, &[]);

    // The run must fail on the NEXT precondition (the connection file), which is
    // what an unsupervised launch without arguments is supposed to hit. Reaching
    // it proves the logger initialized under the fallback id.
    assert!(
        output.contains("missing required --subc connection file path"),
        "an unsupervised launch must reach the connection-file check; got: {output}"
    );
    assert!(
        !output.contains("initialize fleet logger"),
        "the logger must not refuse an unsupervised launch; got: {output}"
    );

    // And it must have actually written a segment, rooted at the default id.
    let segments = home.join("data/cortexkit/synapse/logs");
    let written = std::fs::read_dir(&segments)
        .map(|entries| entries.flatten().count())
        .unwrap_or(0);
    assert!(
        written > 0,
        "the logger should have opened a segment under {}",
        segments.display()
    );
}

#[test]
fn supervised_launch_without_an_id_still_refuses_by_name() {
    let home = isolated_home("supervised");
    let output = launch(&home, &[("SUBC_LAUNCH_NONCE", "test-nonce")]);

    // The fallback must not leak into supervision: a daemon-spawned launch that
    // lost its id is a broken contract, not a local run.
    assert!(
        output.contains("SUBC_MODULE_ID is required when a launch nonce is present"),
        "a supervised launch missing its id must refuse by name; got: {output}"
    );
}

/// The daemon hands the launch nonce over on descriptor 3, named by
/// `SUBC_LAUNCH_NONCE_FD`, and will stop setting the `SUBC_LAUNCH_NONCE` copy.
/// A launch carrying the nonce only on the descriptor is still supervised, so
/// it must hit the same refusal. A module that checked only the environment
/// would take it for a local run and fall back to the default id.
#[cfg(unix)]
#[test]
fn supervised_launch_with_the_nonce_only_on_fd_3_still_refuses_by_name() {
    use subc_os::launch_nonce::{LaunchNonceHandoff, LAUNCH_NONCE_FD_ENV};

    let home = isolated_home("supervised-fd");
    let handoff = LaunchNonceHandoff::new("test-nonce").expect("make the nonce pipe");
    let fd_env = handoff.fd_env_value();
    let output = launch_command(
        &home,
        &[(LAUNCH_NONCE_FD_ENV, fd_env.as_str())],
        |command| handoff.install_last(command),
    );

    assert!(
        output.contains("SUBC_MODULE_ID is required when a launch nonce is present"),
        "a launch with the nonce only on descriptor 3 must refuse by name; got: {output}"
    );
}
