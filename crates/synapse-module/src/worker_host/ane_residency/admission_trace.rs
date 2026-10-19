//! Opt-in, payload-free timing for development stress reports. Detached shape
//! exchanges inherit the requesting task's identity through their final reply.
use super::*;
use std::sync::atomic::AtomicU64;

pub(super) struct Timeline {
    start: std::time::Instant,
    events: Mutex<Vec<serde_json::Value>>,
}

impl Timeline {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            start: std::time::Instant::now(),
            events: Mutex::new(Vec::new()),
        })
    }

    pub(super) fn snapshot(&self) -> Vec<serde_json::Value> {
        self.events.lock().unwrap().clone()
    }

    pub(super) fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

#[derive(Clone)]
pub(super) struct Context {
    timeline: Arc<Timeline>,
    request_id: String,
    ticket: Arc<AtomicU64>,
}

impl Context {
    pub(super) fn new(timeline: Arc<Timeline>, request_id: String) -> Self {
        Self {
            timeline,
            request_id,
            ticket: Arc::new(AtomicU64::new(u64::MAX)),
        }
    }
}

tokio::task_local! { static TRACE: Context; }

pub(super) fn current() -> Option<Context> {
    TRACE.try_with(Clone::clone).ok()
}

pub(super) async fn scope<F: Future>(context: Option<Context>, future: F) -> F::Output {
    match context {
        Some(context) => TRACE.scope(context, future).await,
        None => future.await,
    }
}

pub(super) fn queue(ticket: u64, model: &str, shape: usize) {
    let _ = TRACE.try_with(|context| context.ticket.store(ticket, Ordering::Relaxed));
    event("queue_entry", model, shape);
}

pub(super) fn event(name: &str, model: &str, shape: usize) {
    let _ = TRACE.try_with(|context| {
        // Timestamp under the same lock as insertion to preserve event order
        // when several worker replies arrive on different executor threads.
        let mut events = context.timeline.events.lock().unwrap();
        let ticket = context.ticket.load(Ordering::Relaxed);
        let mut entry = serde_json::json!({
            "event": name, "at_ms": context.timeline.elapsed_ms(),
            "request_id": context.request_id, "model": model, "shape": shape,
        });
        if ticket != u64::MAX {
            entry["ticket"] = ticket.into();
        }
        events.push(entry);
    });
}

pub(super) fn rpc_event(name: &str, request: &WorkerRequest) {
    match request {
        WorkerRequest::AneAdmitShape {
            model_ref, shape, ..
        }
        | WorkerRequest::AneEvictShape {
            model_ref, shape, ..
        } => event(name, model_ref, *shape),
        _ => (),
    }
}

pub(super) struct Rpc<'a>(&'a WorkerRequest);
impl<'a> Rpc<'a> {
    pub(super) fn start(request: &'a WorkerRequest) -> Self {
        rpc_event("rpc_start", request);
        Self(request)
    }
}
impl Drop for Rpc<'_> {
    fn drop(&mut self) {
        rpc_event("rpc_end", self.0);
    }
}
