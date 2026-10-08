#![cfg(windows)]
use std::{
    ffi::c_void,
    fs::File,
    os::windows::io::{AsRawHandle, FromRawHandle},
    process::{Child, Command},
    time::{Duration, Instant},
};
use synapse_core::{
    worker_framing_sync::{read_json_frame, write_frame, write_json_frame},
    WorkerHello, WorkerHelloAck, WorkerRequest, WorkerResponse, DEFAULT_MAX_FRAME_BYTES,
    WORKER_PROTOCOL_VERSION,
};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateNamedPipeW(
        name: *const u16,
        open_mode: u32,
        pipe_mode: u32,
        max_instances: u32,
        out_size: u32,
        in_size: u32,
        timeout: u32,
        security: *mut c_void,
    ) -> *mut c_void;
    fn ConnectNamedPipe(pipe: *mut c_void, overlapped: *mut c_void) -> i32;
    fn SetNamedPipeHandleState(
        pipe: *mut c_void,
        mode: *const u32,
        max_collection: *const u32,
        collection_timeout: *const u32,
    ) -> i32;
}
struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn windows_hosted_worker_hello_ping_unsupported_rerank_ping() {
    let name = format!(r"\\.\pipe\cuda-hosted-{}", std::process::id());
    let wide: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
    // A nonblocking byte pipe lets the test observe an early loader exit while
    // waiting for the worker to connect; no CUDA library is loaded by the test.
    let raw = unsafe {
        CreateNamedPipeW(
            wide.as_ptr(),
            3,
            1,
            1,
            65536,
            65536,
            0,
            std::ptr::null_mut(),
        )
    };
    assert_ne!(raw as isize, -1, "{}", std::io::Error::last_os_error());
    let mut stream = unsafe { File::from_raw_handle(raw) };
    let mut worker = Worker(
        Command::new(
            synapse_core::dev_binary::ckdev_binary(
                env!("CARGO_BIN_EXE_ck-synapse-worker-cuda"),
                std::env::temp_dir().join(format!("cuda-hosted-windows-{}", std::process::id())),
            )
            .unwrap(),
        )
        .args(["--pipe", &name, "--nonce", "hosted-windows"])
        .spawn()
        .unwrap(),
    );
    let start = Instant::now();
    loop {
        if unsafe { ConnectNamedPipe(stream.as_raw_handle(), std::ptr::null_mut()) } != 0 {
            break;
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(535) {
            break;
        }
        assert!(
            worker.0.try_wait().unwrap().is_none(),
            "worker exited before HELLO (possible load-time CUDA import)"
        );
        assert!(
            start.elapsed() < Duration::from_secs(15),
            "worker did not connect: {error}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let mode = 0;
    assert_ne!(
        unsafe {
            SetNamedPipeHandleState(
                stream.as_raw_handle(),
                &mode,
                std::ptr::null(),
                std::ptr::null(),
            )
        },
        0
    );
    let hello: WorkerHello = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
    assert_eq!(hello.nonce, "hosted-windows");
    assert!(hello.kernel_revision.is_some());
    assert_eq!(
        hello.manifest_digest,
        Some(synapse_engine_cuda::manifest::manifest_digest())
    );
    write_json_frame(
        &mut stream,
        &WorkerHelloAck {
            v: WORKER_PROTOCOL_VERSION,
            accept: true,
            max_frame: DEFAULT_MAX_FRAME_BYTES,
        },
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    let ping = |stream: &mut File, id: &str| {
        write_json_frame(
            stream,
            &WorkerRequest::Ping { req_id: id.into() },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        assert!(
            matches!(read_json_frame::<_,WorkerResponse>(stream,DEFAULT_MAX_FRAME_BYTES).unwrap(),WorkerResponse::Pong {req_id,..} if req_id==id)
        );
    };
    ping(&mut stream, "before");
    write_json_frame(
        &mut stream,
        &WorkerRequest::Rerank {
            req_id: "unsupported".into(),
            model_ref: "none".into(),
            query_n_tokens: 1,
            candidates: vec![],
        },
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    write_frame(&mut stream, &42i32.to_le_bytes(), DEFAULT_MAX_FRAME_BYTES).unwrap();
    assert!(
        matches!(read_json_frame::<_,WorkerResponse>(&mut stream,DEFAULT_MAX_FRAME_BYTES).unwrap(),WorkerResponse::Err {code,..} if code=="unsupported_request")
    );
    ping(&mut stream, "after");
    write_json_frame(
        &mut stream,
        &WorkerRequest::Shutdown {},
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    assert!(worker.0.wait().unwrap().success());
}
