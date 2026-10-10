//! A loopback-only, redirect-free mirror for the candidate's normal downloader.
use crate::refuse;
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
use synapse_certify::Result;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::{JoinHandle, JoinSet},
};

pub(crate) struct Mirror {
    pub endpoint: String,
    task: JoinHandle<()>,
}
impl Drop for Mirror {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Mirror {
    pub async fn start(root: &Path) -> Result<Self> {
        let root = std::fs::canonicalize(root).map_err(|e| refuse(format!("mirror: {e}")))?;
        if !root.is_dir() {
            return Err(refuse("mirror must be a directory"));
        }
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| refuse(e.to_string()))?;
        let endpoint = format!(
            "http://{}",
            listener.local_addr().map_err(|e| refuse(e.to_string()))?
        );
        let task = tokio::spawn(async move {
            let mut clients = JoinSet::new();
            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { break };
                        let root = root.clone();
                        clients.spawn(async move {
                            let _ = tokio::time::timeout(Duration::from_secs(120), serve(stream, &root)).await;
                        });
                    }
                    _ = clients.join_next(), if !clients.is_empty() => {}
                }
            }
        });
        Ok(Self { endpoint, task })
    }
}

fn path(root: &Path, uri: &str) -> Option<PathBuf> {
    let encoded = uri.strip_prefix('/')?.split('?').next()?;
    let mut bytes = Vec::new();
    let mut chars = encoded.as_bytes().iter().copied();
    while let Some(c) = chars.next() {
        bytes.push(if c == b'%' {
            let hi = (chars.next()? as char).to_digit(16)?;
            let lo = (chars.next()? as char).to_digit(16)?;
            (hi * 16 + lo) as u8
        } else {
            c
        });
    }
    let decoded = String::from_utf8(bytes).ok()?;
    if decoded.contains(['\\', '\0'])
        || decoded
            .split('/')
            .any(|s| s == ".." || s == "." || s.contains(':'))
    {
        return None;
    }
    let file = std::fs::canonicalize(root.join(decoded)).ok()?;
    (file.starts_with(root) && file.is_file()).then_some(file)
}

async fn serve(mut stream: TcpStream, root: &Path) -> std::io::Result<()> {
    let mut header = Vec::new();
    loop {
        let byte = stream.read_u8().await?;
        header.push(byte);
        if header.ends_with(b"\r\n\r\n") {
            break;
        }
        if header.len() >= 16384 {
            return Ok(());
        }
    }
    let request = String::from_utf8_lossy(&header);
    let mut first = request.lines().next().unwrap_or("").split_whitespace();
    let file = if first.next() == Some("GET") {
        first.next().and_then(|uri| path(root, uri))
    } else {
        None
    };
    let Some(file) = file else {
        stream
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await?;
        return Ok(());
    };
    let file = tokio::fs::File::open(file).await?;
    let length = file.metadata().await?.len();
    stream
        .write_all(
            format!("HTTP/1.1 200 OK\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .await?;
    tokio::io::copy(&mut file.take(length), &mut stream).await?;
    stream.shutdown().await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn serves_hf_layout_with_length_and_refuses_traversal() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!(".live/mirror-test-{}", std::process::id()));
        std::fs::create_dir_all(root.join("org/repo/resolve/pin")).unwrap();
        std::fs::write(root.join("org/repo/resolve/pin/model.bin"), b"weights").unwrap();
        let mirror = Mirror::start(&root).await.unwrap();
        for (uri, expected) in [
            ("/org/repo/resolve/pin/model.bin", "200 OK"),
            ("/%2e%2e/model.bin", "404 Not Found"),
            ("/missing", "404 Not Found"),
        ] {
            let mut stream = TcpStream::connect(mirror.endpoint.strip_prefix("http://").unwrap())
                .await
                .unwrap();
            stream
                .write_all(format!("GET {uri} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes())
                .await
                .unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).await.unwrap();
            assert!(response.contains(expected), "{response}");
            if expected == "200 OK" {
                assert!(response.ends_with("Content-Length: 7\r\nConnection: close\r\n\r\nweights"));
            }
        }
        drop(mirror);
        std::fs::remove_dir_all(root).unwrap();
    }
}
