use std::io;

use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub async fn read_frame<S>(stream: &mut S, max_frame: u32) -> io::Result<Vec<u8>>
where
    S: AsyncRead + Unpin,
{
    let mut len_bytes = [0_u8; 4];
    stream.read_exact(&mut len_bytes).await?;
    let len = u32::from_le_bytes(len_bytes);
    if len > max_frame {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {max_frame}"),
        ));
    }
    let mut frame = vec![0_u8; len as usize];
    stream.read_exact(&mut frame).await?;
    Ok(frame)
}

pub async fn write_frame<S>(stream: &mut S, bytes: &[u8], max_frame: u32) -> io::Result<()>
where
    S: AsyncWrite + Unpin,
{
    let len = u32::try_from(bytes.len()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "frame too large for u32 length")
    })?;
    if len > max_frame {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {max_frame}"),
        ));
    }
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(bytes).await?;
    stream.flush().await?;
    Ok(())
}

pub async fn read_json_frame<S, T>(stream: &mut S, max_frame: u32) -> io::Result<T>
where
    S: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let frame = read_frame(stream, max_frame).await?;
    serde_json::from_slice(&frame).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("worker JSON decode: {error}"),
        )
    })
}

pub async fn write_json_frame<S, T>(stream: &mut S, value: &T, max_frame: u32) -> io::Result<()>
where
    S: AsyncWrite + Unpin,
    T: Serialize,
{
    let bytes = serde_json::to_vec(value).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("worker JSON encode: {error}"),
        )
    })?;
    write_frame(stream, &bytes, max_frame).await
}
