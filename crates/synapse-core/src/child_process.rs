use std::process::Command;

/// Keep the module's daemon credential out of children, which must not be able
/// to authenticate as the module. The descriptor hint is also stale after the
/// SDK consumes the nonce and closes its handoff descriptor.
pub fn without_launch_nonce(mut command: Command) -> Command {
    remove_launch_nonce(&mut command);
    command
}

/// Strip the same daemon credential from asynchronous child commands.
pub fn without_launch_nonce_tokio(mut command: tokio::process::Command) -> tokio::process::Command {
    remove_launch_nonce(command.as_std_mut());
    command
}

fn remove_launch_nonce(command: &mut Command) {
    command.env_remove(subc_os::launch_nonce::LAUNCH_NONCE_ENV);
    command.env_remove(subc_os::launch_nonce::LAUNCH_NONCE_FD_ENV);
}
