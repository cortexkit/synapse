use std::io;
use std::time::Duration;

use crate::worker_framing_sync::{read_json_frame, write_json_frame};
use crate::worker_protocol::{WorkerHello, WorkerHelloAck, WORKER_PROTOCOL_VERSION};

/// Blocking worker-side connect + HELLO on a Windows named pipe.
pub fn connect_and_handshake(
    pipe_name: &str,
    hello: &WorkerHello,
    max_frame: u32,
) -> io::Result<(std::fs::File, u32)> {
    use std::fs::OpenOptions;
    use std::thread;
    use std::time::Instant;

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut last_error = io::Error::new(io::ErrorKind::NotFound, "pipe not ready");
    let mut client = loop {
        match OpenOptions::new().read(true).write(true).open(pipe_name) {
            Ok(file) => break file,
            Err(error) if Instant::now() < deadline => {
                last_error = error;
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(error),
        }
    };
    write_json_frame(&mut client, hello, max_frame)?;
    let ack: WorkerHelloAck = read_json_frame(&mut client, max_frame)?;
    if ack.v != WORKER_PROTOCOL_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("module replied with protocol v{}", ack.v),
        ));
    }
    if !ack.accept {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "module rejected worker handshake",
        ));
    }
    let _ = last_error;
    let negotiated = max_frame.min(ack.max_frame);
    Ok((client, negotiated))
}
