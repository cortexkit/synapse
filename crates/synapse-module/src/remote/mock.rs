use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    io,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
    time::sleep,
};
use url::Url;

const DEFAULT_DIMENSIONS: usize = 3;
const MAX_REQUEST_HEADERS: usize = 64 * 1024;

#[derive(Clone, Debug)]
pub enum MockBehavior {
    Ok,
    RateLimited { retry_after: Option<String> },
    ServerError,
    Hang { duration: Duration },
    WrongDimensions,
    DuplicateIndex,
    ShortCount,
    NonFiniteValue,
    OversizedBody { bytes: usize },
    Redirect { location: String },
    SilentTruncateEcho { max_chars: usize, dimensions: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedRequest {
    pub method: String,
    pub route: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

pub struct MockProvider {
    address: SocketAddr,
    scripts: Arc<Mutex<HashMap<String, VecDeque<MockBehavior>>>>,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    accept_task: JoinHandle<()>,
}

impl MockProvider {
    pub async fn start() -> io::Result<Self> {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
        let address = listener.local_addr()?;
        let scripts = Arc::new(Mutex::new(HashMap::new()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let task_scripts = Arc::clone(&scripts);
        let task_requests = Arc::clone(&requests);
        let accept_task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let scripts = Arc::clone(&task_scripts);
                let requests = Arc::clone(&task_requests);
                tokio::spawn(async move {
                    let _ = serve_connection(stream, scripts, requests).await;
                });
            }
        });

        Ok(Self {
            address,
            scripts,
            requests,
            accept_task,
        })
    }

    pub fn enqueue(&self, route: &str, behavior: MockBehavior) {
        self.scripts
            .lock()
            .expect("mock provider scripts lock poisoned")
            .entry(route.to_string())
            .or_default()
            .push_back(behavior);
    }

    pub fn enqueue_all(&self, route: &str, behaviors: impl IntoIterator<Item = MockBehavior>) {
        self.scripts
            .lock()
            .expect("mock provider scripts lock poisoned")
            .entry(route.to_string())
            .or_default()
            .extend(behaviors);
    }

    pub fn url(&self, route: &str) -> Url {
        Url::parse(&format!("http://{}{}", self.address, route))
            .expect("mock provider URL must be valid")
    }

    pub fn requests_for(&self, route: &str) -> Vec<RecordedRequest> {
        self.requests
            .lock()
            .expect("mock provider requests lock poisoned")
            .iter()
            .filter(|request| request.route == route)
            .cloned()
            .collect()
    }
}

impl Drop for MockProvider {
    fn drop(&mut self) {
        self.accept_task.abort();
    }
}

async fn serve_connection(
    mut stream: TcpStream,
    scripts: Arc<Mutex<HashMap<String, VecDeque<MockBehavior>>>>,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
) -> io::Result<()> {
    let Some(request) = read_request(&mut stream).await? else {
        return Ok(());
    };
    let behavior = scripts
        .lock()
        .expect("mock provider scripts lock poisoned")
        .get_mut(&request.route)
        .and_then(VecDeque::pop_front);
    requests
        .lock()
        .expect("mock provider requests lock poisoned")
        .push(request.clone());

    let response = match behavior {
        Some(behavior) => response_for(behavior, &request).await,
        None => MockResponse::json(404, json!({"error": "no scripted response"})),
    };
    write_response(&mut stream, response).await
}

async fn read_request(stream: &mut TcpStream) -> io::Result<Option<RecordedRequest>> {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Ok(None);
        }
        bytes.extend_from_slice(&chunk[..read]);
        if bytes.len() > MAX_REQUEST_HEADERS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mock request headers exceeded limit",
            ));
        }
        if let Some(offset) = find_subslice(&bytes, b"\r\n\r\n") {
            break offset + 4;
        }
    };

    let header_text = std::str::from_utf8(&bytes[..header_end]).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("mock request headers were not UTF-8: {error}"),
        )
    })?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or_default().to_string();
    let target = request_parts.next().unwrap_or_default();
    let route = target.split('?').next().unwrap_or(target).to_string();
    let mut headers = BTreeMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let content_length = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let total_length = header_end.checked_add(content_length).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "mock request length overflow")
    })?;
    while bytes.len() < total_length {
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "mock request body ended early",
            ));
        }
        bytes.extend_from_slice(&chunk[..read]);
    }

    Ok(Some(RecordedRequest {
        method,
        route,
        headers,
        body: bytes[header_end..total_length].to_vec(),
    }))
}

async fn response_for(behavior: MockBehavior, request: &RecordedRequest) -> MockResponse {
    match behavior {
        MockBehavior::Ok => embedding_response(request, ResponseShape::Normal),
        MockBehavior::RateLimited { retry_after } => {
            let mut response = MockResponse::json(429, json!({"error": "rate limited"}));
            if let Some(value) = retry_after {
                response.headers.push(("Retry-After", value));
            }
            response
        }
        MockBehavior::ServerError => {
            MockResponse::json(500, json!({"error": "scripted server error"}))
        }
        MockBehavior::Hang { duration } => {
            sleep(duration).await;
            embedding_response(request, ResponseShape::Normal)
        }
        MockBehavior::WrongDimensions => {
            embedding_response(request, ResponseShape::WrongDimensions)
        }
        MockBehavior::DuplicateIndex => embedding_response(request, ResponseShape::DuplicateIndex),
        MockBehavior::ShortCount => embedding_response(request, ResponseShape::ShortCount),
        MockBehavior::NonFiniteValue => non_finite_response(request),
        MockBehavior::OversizedBody { bytes } => MockResponse {
            status: 200,
            headers: vec![("Content-Type", "application/json".to_string())],
            body: vec![b'x'; bytes],
        },
        MockBehavior::Redirect { location } => MockResponse {
            status: 302,
            headers: vec![("Location", location)],
            body: Vec::new(),
        },
        MockBehavior::SilentTruncateEcho {
            max_chars,
            dimensions,
        } => silent_truncate_echo_response(request, max_chars, dimensions),
    }
}

#[derive(Clone, Copy)]
enum ResponseShape {
    Normal,
    WrongDimensions,
    DuplicateIndex,
    ShortCount,
}

#[derive(Deserialize)]
struct IncomingEmbeddingRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    input: Vec<String>,
    dimensions: Option<usize>,
}

fn embedding_response(request: &RecordedRequest, shape: ResponseShape) -> MockResponse {
    let incoming = incoming_embedding_request(request);
    let expected_dimensions = incoming.dimensions.unwrap_or(DEFAULT_DIMENSIONS);
    let dimensions = match shape {
        ResponseShape::WrongDimensions => expected_dimensions.saturating_add(1),
        _ => expected_dimensions,
    };
    let count = match shape {
        ResponseShape::ShortCount => incoming.input.len().saturating_sub(1),
        _ => incoming.input.len(),
    };
    let data = (0..count)
        .map(|index| {
            let response_index = match shape {
                ResponseShape::DuplicateIndex => 0,
                _ => index,
            };
            json!({
                "object": "embedding",
                "index": response_index,
                "embedding": vector_for(index, dimensions),
            })
        })
        .collect::<Vec<_>>();
    MockResponse::json(
        200,
        json!({
            "object": "list",
            "data": data,
            "model": incoming.model,
            "usage": {
                "prompt_tokens": incoming.input.len(),
                "total_tokens": incoming.input.len(),
            }
        }),
    )
}

fn non_finite_response(request: &RecordedRequest) -> MockResponse {
    let incoming = incoming_embedding_request(request);
    let dimensions = incoming.dimensions.unwrap_or(DEFAULT_DIMENSIONS).max(1);
    let mut data = Vec::with_capacity(incoming.input.len());
    for index in 0..incoming.input.len() {
        let vector = if index == 0 {
            let remaining = std::iter::repeat_n("0.0", dimensions - 1)
                .collect::<Vec<_>>()
                .join(",");
            if remaining.is_empty() {
                "[1e999]".to_string()
            } else {
                format!("[1e999,{remaining}]")
            }
        } else {
            serde_json::to_string(&vector_for(index, dimensions))
                .expect("finite mock vector must serialize")
        };
        data.push(format!("{{\"index\":{index},\"embedding\":{vector}}}"));
    }
    let model = serde_json::to_string(&incoming.model).expect("mock model must serialize");
    let body = format!(
        "{{\"data\":[{}],\"model\":{model},\"usage\":{{\"prompt_tokens\":{},\"total_tokens\":{}}}}}",
        data.join(","),
        incoming.input.len(),
        incoming.input.len()
    )
    .into_bytes();
    MockResponse {
        status: 200,
        headers: vec![("Content-Type", "application/json".to_string())],
        body,
    }
}

fn silent_truncate_echo_response(
    request: &RecordedRequest,
    max_chars: usize,
    dimensions: usize,
) -> MockResponse {
    let incoming = incoming_embedding_request(request);
    let truncated = incoming
        .input
        .iter()
        .map(|input| input.chars().take(max_chars).collect::<String>())
        .collect::<Vec<_>>();
    let data = truncated
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let mut embedding = vec![0.0; dimensions];
            if let Some(first) = embedding.first_mut() {
                *first = value.chars().count() as f64;
            }
            json!({"index": index, "embedding": embedding})
        })
        .collect::<Vec<_>>();
    MockResponse::json(
        200,
        json!({
            "data": data,
            "model": format!("silent-truncate-echo:{}", truncated.join("|")),
            "usage": {
                "prompt_tokens": truncated.iter().map(|value| value.chars().count()).sum::<usize>(),
                "total_tokens": truncated.iter().map(|value| value.chars().count()).sum::<usize>(),
            }
        }),
    )
}

fn incoming_embedding_request(request: &RecordedRequest) -> IncomingEmbeddingRequest {
    serde_json::from_slice(&request.body).unwrap_or(IncomingEmbeddingRequest {
        model: "mock-model".to_string(),
        input: Vec::new(),
        dimensions: None,
    })
}

fn vector_for(index: usize, dimensions: usize) -> Vec<f64> {
    (0..dimensions)
        .map(|coordinate| index as f64 + coordinate as f64 / 10.0)
        .collect()
}

struct MockResponse {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: Vec<u8>,
}

impl MockResponse {
    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            headers: vec![("Content-Type", "application/json".to_string())],
            body: serde_json::to_vec(&value).expect("mock JSON response must serialize"),
        }
    }
}

async fn write_response(stream: &mut TcpStream, response: MockResponse) -> io::Result<()> {
    let reason = match response.status {
        200 => "OK",
        302 => "Found",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "Mock Response",
    };
    let mut head = format!(
        "HTTP/1.1 {} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
        response.status,
        response.body.len()
    );
    for (name, value) in response.headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(&value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(&response.body).await?;
    stream.shutdown().await
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::openai_compat::parse_embedding_response;
    use reqwest::StatusCode;

    fn request_body(inputs: &[&str], dimensions: usize) -> Value {
        json!({"model": "mocked", "input": inputs, "dimensions": dimensions})
    }

    #[tokio::test]
    async fn scripts_are_fifo_and_isolated_per_route() {
        let provider = MockProvider::start().await.unwrap();
        provider.enqueue_all(
            "/embeddings",
            [
                MockBehavior::ServerError,
                MockBehavior::RateLimited {
                    retry_after: Some("7".to_string()),
                },
                MockBehavior::Ok,
            ],
        );
        provider.enqueue("/other", MockBehavior::Ok);
        let client = reqwest::Client::new();

        let first = client
            .post(provider.url("/embeddings"))
            .json(&request_body(&["a"], 2))
            .send()
            .await
            .unwrap();
        let second = client
            .post(provider.url("/embeddings"))
            .json(&request_body(&["b"], 2))
            .send()
            .await
            .unwrap();
        let other = client
            .post(provider.url("/other"))
            .json(&request_body(&["c"], 2))
            .send()
            .await
            .unwrap();
        let third = client
            .post(provider.url("/embeddings"))
            .json(&request_body(&["d"], 2))
            .send()
            .await
            .unwrap();

        assert_eq!(first.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(second.headers()["retry-after"], "7");
        assert_eq!(other.status(), StatusCode::OK);
        assert_eq!(third.status(), StatusCode::OK);
        assert_eq!(provider.requests_for("/embeddings").len(), 3);
        assert_eq!(provider.requests_for("/other").len(), 1);
    }

    #[tokio::test]
    async fn protocol_fault_behaviors_emit_the_requested_shapes() {
        let provider = MockProvider::start().await.unwrap();
        provider.enqueue_all(
            "/embeddings",
            [
                MockBehavior::WrongDimensions,
                MockBehavior::DuplicateIndex,
                MockBehavior::ShortCount,
                MockBehavior::NonFiniteValue,
            ],
        );
        let client = reqwest::Client::new();
        let mut bodies = Vec::new();
        for _ in 0..4 {
            bodies.push(
                client
                    .post(provider.url("/embeddings"))
                    .json(&request_body(&["a", "b"], 2))
                    .send()
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap(),
            );
        }

        let wrong = parse_embedding_response(&bodies[0]).unwrap();
        assert_eq!(wrong.data[0].embedding.len(), 3);
        let duplicate = parse_embedding_response(&bodies[1]).unwrap();
        assert_eq!(duplicate.data[0].index, duplicate.data[1].index);
        let short = parse_embedding_response(&bodies[2]).unwrap();
        assert_eq!(short.data.len(), 1);
        assert!(parse_embedding_response(&bodies[3]).is_err());
    }

    #[tokio::test]
    async fn silent_truncate_echo_accepts_over_limit_input_and_discloses_truncation() {
        let provider = MockProvider::start().await.unwrap();
        provider.enqueue(
            "/embeddings",
            MockBehavior::SilentTruncateEcho {
                max_chars: 4,
                dimensions: 2,
            },
        );

        let response = reqwest::Client::new()
            .post(provider.url("/embeddings"))
            .json(&request_body(&["abcdefgh"], 2))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let parsed = parse_embedding_response(&response.bytes().await.unwrap()).unwrap();
        assert_eq!(parsed.model, "silent-truncate-echo:abcd");
        assert_eq!(parsed.data[0].embedding, vec![4.0, 0.0]);
    }
}
