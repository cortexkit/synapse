use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub mod windows_client;

#[cfg(unix)]
pub use unix::{
    accept_worker_handshake, bind_listener, handshake_on_stream, prepare_listener, read_json,
    read_raw, write_json, write_raw, TransportError, WorkerTransportStream,
};
#[cfg(windows)]
pub use windows::{
    accept_worker_handshake, bind_listener, handshake_on_stream, prepare_listener, read_json,
    read_raw, write_json, write_raw, TransportError, WorkerTransportStream,
};

/// Short stable digest of `worker_id` for IPC endpoint names (SUN_LEN-safe on Unix).
pub fn worker_endpoint_digest(worker_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(worker_id.as_bytes());
    hex::encode(&hasher.finalize()[..8])
}

/// Module-owned Unix socket path: `<runtime_dir>/wk-<digest>.sock`.
pub fn worker_socket_path(runtime_dir: &Path, worker_id: &str) -> PathBuf {
    let digest = worker_endpoint_digest(worker_id);
    runtime_dir.join(format!("wk-{digest}.sock"))
}

/// Windows named pipe for the same worker id: `\\.\pipe\synapse-<digest>`.
pub fn worker_pipe_name(worker_id: &str) -> String {
    format!(r"\\.\pipe\synapse-{}", worker_endpoint_digest(worker_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_stable_and_short() {
        let a = worker_endpoint_digest("synapse-llama-test");
        let b = worker_endpoint_digest("synapse-llama-test");
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn socket_path_uses_digest_not_raw_id() {
        let path = worker_socket_path(
            Path::new("/tmp/syn"),
            "very-long-worker-id-that-would-overflow-sun-len",
        );
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("wk-"));
        assert!(name.ends_with(".sock"));
    }

    #[cfg(windows)]
    #[test]
    fn pipe_name_matches_contract() {
        let name = worker_pipe_name("worker-a");
        assert!(name.starts_with(r"\\.\pipe\synapse-"));
        assert_eq!(name, worker_pipe_name("worker-a"));
    }
}
