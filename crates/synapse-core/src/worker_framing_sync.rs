use std::io::{self, Read, Write};

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::{WorkerProgressPhase, WorkerResponse};

/// Serial writer for completed work, sharing the request's terminal-reply stream.
/// Call only after a native completion fence, never from a timer or graph capture.
pub struct ProgressWriter<'a, W> {
    writer: &'a mut W,
    req_id: &'a str,
    max_frame: u32,
    completed: u64,
    failed: bool,
}

impl<'a, W: Write> ProgressWriter<'a, W> {
    pub fn new(writer: &'a mut W, req_id: &'a str, max_frame: u32) -> Self {
        Self {
            writer,
            req_id,
            max_frame,
            completed: 0,
            failed: false,
        }
    }

    pub fn completed(&mut self, phase: WorkerProgressPhase) -> io::Result<()> {
        self.completed = self.completed.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "progress counter overflow")
        })?;
        let result = write_json_frame(
            self.writer,
            &WorkerResponse::Progress {
                req_id: self.req_id.into(),
                completed: self.completed,
                phase,
            },
            self.max_frame,
        );
        self.failed |= result.is_err();
        result
    }

    pub fn failed(&self) -> bool {
        self.failed
    }
}

pub fn read_frame<R: Read>(reader: &mut R, max_frame: u32) -> io::Result<Vec<u8>> {
    let mut len_bytes = [0_u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes);
    if len > max_frame {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {max_frame}"),
        ));
    }
    let mut frame = vec![0_u8; len as usize];
    reader.read_exact(&mut frame)?;
    Ok(frame)
}

pub fn write_frame<W: Write>(writer: &mut W, bytes: &[u8], max_frame: u32) -> io::Result<()> {
    let len = u32::try_from(bytes.len()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "frame too large for u32 length")
    })?;
    if len > max_frame {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {max_frame}"),
        ));
    }
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(bytes)?;
    writer.flush()?;
    Ok(())
}

pub fn read_json_frame<R: Read, T: DeserializeOwned>(
    reader: &mut R,
    max_frame: u32,
) -> io::Result<T> {
    let frame = read_frame(reader, max_frame)?;
    serde_json::from_slice(&frame).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("worker JSON decode: {error}"),
        )
    })
}

pub fn write_json_frame<W: Write, T: Serialize>(
    writer: &mut W,
    value: &T,
    max_frame: u32,
) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("worker JSON encode: {error}"),
        )
    })?;
    write_frame(writer, &bytes, max_frame)
}
