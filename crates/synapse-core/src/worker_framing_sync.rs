use std::io::{self, Read, Write};

use serde::de::DeserializeOwned;
use serde::Serialize;

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
