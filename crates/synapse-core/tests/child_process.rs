use std::process::Command;
use subc_os::launch_nonce::{LAUNCH_NONCE_ENV, LAUNCH_NONCE_FD_ENV};
use synapse_core::without_launch_nonce;

fn environment_command() -> Command {
    #[cfg(windows)]
    let mut command = {
        let mut command = Command::new("cmd");
        command.args(["/c", "set"]);
        command
    };
    #[cfg(not(windows))]
    let mut command = Command::new("/usr/bin/env");
    command.env(LAUNCH_NONCE_ENV, "credential-control");
    command.env(LAUNCH_NONCE_FD_ENV, "3:123");
    command
}

fn environment(mut command: Command) -> String {
    let output = command.output().expect("spawn environment printer");
    assert!(output.status.success());
    String::from_utf8(output.stdout).expect("environment is UTF-8")
}

#[test]
fn child_environment_control_contains_both_launch_nonce_variables() {
    let output = environment(environment_command());
    assert!(output
        .lines()
        .any(|line| line == format!("{LAUNCH_NONCE_ENV}=credential-control")));
    assert!(output
        .lines()
        .any(|line| line == format!("{LAUNCH_NONCE_FD_ENV}=3:123")));
}

#[test]
fn stripped_child_environment_contains_neither_launch_nonce_variable() {
    let output = environment(without_launch_nonce(environment_command()));
    for name in [LAUNCH_NONCE_ENV, LAUNCH_NONCE_FD_ENV] {
        assert!(
            !output
                .lines()
                .any(|line| line.starts_with(&format!("{name}="))),
            "child inherited {name}"
        );
    }
}

#[tokio::test]
async fn stripped_tokio_child_environment_contains_neither_launch_nonce_variable() {
    let command = tokio::process::Command::from(environment_command());
    let output = synapse_core::without_launch_nonce_tokio(command)
        .output()
        .await
        .expect("spawn async environment printer");
    assert!(output.status.success());
    let output = String::from_utf8(output.stdout).expect("environment is UTF-8");
    for name in [LAUNCH_NONCE_ENV, LAUNCH_NONCE_FD_ENV] {
        assert!(
            !output
                .lines()
                .any(|line| line.starts_with(&format!("{name}="))),
            "child inherited {name}"
        );
    }
}
