//! Worker stdout/stderr capture: the crash-log tail, and forwarding of each
//! complete line to the module's log under a per-worker rate limit.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncReadExt};

const STDERR_RING_BYTES: usize = 8 * 1024;

/// The most bytes of one line kept for forwarding. A longer line is cut
/// there and the rest discarded up to its newline, so a worker that writes
/// without newlines (repeated `\r` progress updates, for example) cannot grow
/// the reader's memory or its scanning work. It matches the crash-tail size.
const MAX_LINE_BYTES: usize = STDERR_RING_BYTES;

#[derive(Clone, Debug)]
pub(super) struct LogRing {
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl LogRing {
    pub(super) fn new() -> Self {
        Self {
            bytes: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn push(&self, bytes: &[u8]) {
        let mut guard = self.bytes.lock().expect("stderr ring mutex poisoned");
        guard.extend_from_slice(bytes);
        if guard.len() > STDERR_RING_BYTES {
            let drain = guard.len() - STDERR_RING_BYTES;
            guard.drain(0..drain);
        }
    }

    pub(super) fn tail(&self) -> String {
        let guard = self.bytes.lock().expect("stderr ring mutex poisoned");
        String::from_utf8_lossy(&guard).into_owned()
    }
}

#[derive(Default)]
pub(super) struct WorkerLogContext {
    pub(super) job_id: Option<String>,
}

pub(super) struct WorkerForwardLimiter {
    cap: u32,
    window_started: Instant,
    forwarded: u32,
    dropped: u64,
}

impl WorkerForwardLimiter {
    pub(super) fn new(cap: u32) -> Self {
        Self {
            cap,
            window_started: Instant::now(),
            forwarded: 0,
            dropped: 0,
        }
    }

    fn admit(&mut self, now: Instant) -> Option<u64> {
        if now.duration_since(self.window_started) >= Duration::from_secs(1) {
            self.window_started = now;
            self.forwarded = 0;
        }
        if self.forwarded >= self.cap {
            self.dropped = self.dropped.saturating_add(1);
            return None;
        }
        self.forwarded = self.forwarded.saturating_add(1);
        Some(std::mem::take(&mut self.dropped))
    }
}

/// Splits a byte stream into lines, looking at each byte once and keeping at
/// most [`MAX_LINE_BYTES`] of the line in progress.
#[derive(Default)]
struct LineSplitter {
    line: Vec<u8>,
    truncated: bool,
}

impl LineSplitter {
    /// Feeds `bytes` and calls `emit` with each line they complete, without
    /// its line ending. A cut line ends with a truncation marker.
    fn feed(&mut self, mut bytes: &[u8], mut emit: impl FnMut(&str)) {
        while let Some(newline) = bytes.iter().position(|byte| *byte == b'\n') {
            self.keep(&bytes[..newline]);
            if !self.truncated && self.line.last() == Some(&b'\r') {
                self.line.pop();
            }
            let mut text = String::from_utf8_lossy(&self.line).into_owned();
            if self.truncated {
                text.push_str(" [line truncated]");
            }
            emit(&text);
            self.line.clear();
            self.truncated = false;
            bytes = &bytes[newline + 1..];
        }
        self.keep(bytes);
    }

    fn keep(&mut self, bytes: &[u8]) {
        let room = MAX_LINE_BYTES - self.line.len();
        if bytes.len() > room {
            self.truncated = true;
        }
        self.line.extend_from_slice(&bytes[..bytes.len().min(room)]);
    }
}

pub(super) fn spawn_pipe_reader<R>(
    stream: &'static str,
    mut reader: R,
    ring: LogRing,
    worker_id: String,
    model_id: String,
    context: Arc<Mutex<WorkerLogContext>>,
    limiter: Arc<Mutex<WorkerForwardLimiter>>,
) where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let prefix = format!("[{stream}] ");
        let mut buffer = [0_u8; 512];
        let mut lines = LineSplitter::default();
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) => break,
                Ok(n) => {
                    ring.push(prefix.as_bytes());
                    ring.push(&buffer[..n]);
                    lines.feed(&buffer[..n], |line| {
                        forward_worker_line(&worker_id, stream, &model_id, line, &context, &limiter)
                    });
                }
                Err(_) => break,
            }
        }
    });
}

fn forward_worker_line(
    worker_id: &str,
    stream: &str,
    model_id: &str,
    line: &str,
    context: &Mutex<WorkerLogContext>,
    limiter: &Mutex<WorkerForwardLimiter>,
) {
    let dropped = match limiter.lock() {
        Ok(mut limiter) => match limiter.admit(Instant::now()) {
            Some(dropped) => dropped,
            None => return,
        },
        Err(_) => return,
    };
    let job_id = context
        .lock()
        .ok()
        .and_then(|context| context.job_id.clone());
    match (job_id.as_deref(), dropped) {
        (Some(job_id), 0) => tracing::info!(
            target: "worker",
            worker = worker_id,
            stream,
            model_id,
            job_id,
            "{line}"
        ),
        (Some(job_id), dropped) => tracing::info!(
            target: "worker",
            worker = worker_id,
            stream,
            model_id,
            job_id,
            dropped,
            "{line}"
        ),
        (None, 0) => tracing::info!(
            target: "worker",
            worker = worker_id,
            stream,
            model_id,
            "{line}"
        ),
        (None, dropped) => tracing::info!(
            target: "worker",
            worker = worker_id,
            stream,
            model_id,
            dropped,
            "{line}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(splitter: &mut LineSplitter, bytes: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        for chunk in bytes.chunks(512) {
            splitter.feed(chunk, |line| lines.push(line.to_owned()));
        }
        lines
    }

    #[test]
    fn a_line_without_newline_keeps_bounded_memory_and_is_cut() {
        let mut splitter = LineSplitter::default();
        let progress = b"\rprogress 42%".repeat(100_000);
        assert!(feed_all(&mut splitter, &progress).is_empty());
        assert!(
            splitter.line.len() <= MAX_LINE_BYTES,
            "kept {} bytes of an unfinished line",
            splitter.line.len()
        );
        let lines = feed_all(&mut splitter, b"\nnext\r\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].len(), MAX_LINE_BYTES + " [line truncated]".len());
        assert!(lines[0].ends_with(" [line truncated]"));
        assert_eq!(lines[1], "next");
    }

    #[test]
    fn lines_split_across_reads_are_joined() {
        let mut splitter = LineSplitter::default();
        let mut lines = Vec::new();
        for chunk in [&b"one\ntw"[..], b"o\r", b"\nthree"] {
            splitter.feed(chunk, |line| lines.push(line.to_owned()));
        }
        assert_eq!(lines, ["one", "two"]);
        assert_eq!(splitter.line, b"three");
    }
}
