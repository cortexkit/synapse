use super::*;

pub mod progress;
pub mod quarantine;

impl WorkerHost {
    pub fn advance_clock_for_test(&mut self, elapsed: Duration) {
        self.clock = Some(self.clock.unwrap() + elapsed);
    }

    pub fn worker_generation_for_test(&self) -> u64 {
        self.connection.as_ref().unwrap().worker_generation
    }

    pub async fn shutdown_for_test(&mut self) {
        self.kill_current().await;
    }
}

#[cfg(unix)]
type SeenRequests = Vec<(WorkerRequest, Option<Vec<u8>>)>;

/// Connects `host` to an in-process mock worker that answers each request
/// with `respond`. The returned task yields every request (and its raw
/// frame) once the host drops the connection.
#[cfg(unix)]
pub fn attach_mock_worker<F>(
    host: &mut WorkerHost,
    respond: F,
) -> tokio::task::JoinHandle<SeenRequests>
where
    F: Fn(&WorkerRequest, Option<&[u8]>) -> (WorkerResponse, Option<Vec<u8>>) + Send + 'static,
{
    attach_mock_worker_frames(host, move |request, raw| vec![respond(request, raw)])
}

#[cfg(unix)]
pub fn attach_mock_worker_frames<F>(
    host: &mut WorkerHost,
    respond: F,
) -> tokio::task::JoinHandle<SeenRequests>
where
    F: Fn(&WorkerRequest, Option<&[u8]>) -> Vec<(WorkerResponse, Option<Vec<u8>>)> + Send + 'static,
{
    let (module, mut worker) = tokio::net::UnixStream::pair().unwrap();
    // The mock below speaks the protocol in-process; the host still owns a
    // child process per connection, so give it a harmless one.
    let child = synapse_core::without_launch_nonce_tokio(Command::new("sleep"))
        .arg("30")
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    host.connection = Some(WorkerConnection {
        stream: module,
        child,
        logs: LogRing::new(),
        worker_generation: 1,
        packed: None,
        max_frame: synapse_core::DEFAULT_MAX_FRAME_BYTES,
        unusable: false,
    });
    tokio::spawn(async move {
        let mut seen = Vec::new();
        while let Ok(request) =
            read_json::<WorkerRequest, _>(&mut worker, DEFAULT_MAX_FRAME_BYTES).await
        {
            let raw = if request.carries_raw_frame() {
                Some(
                    read_raw(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                        .await
                        .unwrap(),
                )
            } else {
                None
            };
            for (response, raw_out) in respond(&request, raw.as_deref()) {
                write_json(&mut worker, &response, DEFAULT_MAX_FRAME_BYTES)
                    .await
                    .unwrap();
                if let Some(raw_out) = raw_out {
                    write_raw(&mut worker, &raw_out, DEFAULT_MAX_FRAME_BYTES)
                        .await
                        .unwrap();
                }
            }
            seen.push((request, raw));
        }
        seen
    })
}
