use std::{
    io,
    net::{IpAddr, Ipv6Addr, SocketAddr},
    time::Duration,
};

use reqwest::{
    header::HeaderMap, redirect::Policy, Client, Method, Request, RequestBuilder, StatusCode, Url,
};
use thiserror::Error;
use tokio::{net::TcpStream, time::timeout};
use url::Host;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EndpointSecurity {
    LoopbackAuthNone,
    ProviderManagedAuth,
}

#[derive(Clone, Debug)]
pub(super) struct GatewayHttpClient {
    inner: Client,
    connect_timeout: Duration,
    response_body_limit: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CappedResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

#[derive(Debug, Error)]
pub(super) enum GatewayClientError {
    #[error("failed to build gateway HTTP client: {0}")]
    Build(#[source] reqwest::Error),
    #[error("auth-none endpoint host `{host}` is not allowed: {reason}")]
    InvalidLoopbackHost { host: String, reason: String },
    #[error("gateway endpoint URL has no host")]
    MissingHost,
    #[error("gateway endpoint URL has no usable port")]
    MissingPort,
    #[error("loopback peer pre-flight timed out after {timeout:?}")]
    PeerConnectTimeout { timeout: Duration },
    #[error("loopback peer pre-flight failed for {address}: {source}")]
    PeerConnect {
        address: SocketAddr,
        #[source]
        source: io::Error,
    },
    #[error("failed to inspect connected gateway peer: {0}")]
    PeerAddress(#[source] io::Error),
    #[error("connected gateway peer {peer} is not an allowed loopback address")]
    PeerNotLoopback { peer: SocketAddr },
    #[error("gateway request failed: {0}")]
    Request(#[source] reqwest::Error),
    #[error("gateway response body exceeded {limit} byte limit after receiving at least {received} bytes")]
    BodyTooLarge { limit: usize, received: usize },
}

impl GatewayHttpClient {
    pub(super) fn new(
        connect_timeout: Duration,
        read_timeout: Duration,
        response_body_limit: usize,
    ) -> Result<Self, GatewayClientError> {
        let inner = Client::builder()
            .redirect(Policy::none())
            .no_proxy()
            .connect_timeout(connect_timeout)
            .read_timeout(read_timeout)
            .build()
            .map_err(GatewayClientError::Build)?;
        Ok(Self {
            inner,
            connect_timeout,
            response_body_limit,
        })
    }

    pub(super) fn request(&self, method: Method, url: Url) -> RequestBuilder {
        self.inner.request(method, url)
    }

    pub(super) async fn execute(
        &self,
        request: Request,
        security: EndpointSecurity,
    ) -> Result<CappedResponse, GatewayClientError> {
        if security == EndpointSecurity::LoopbackAuthNone {
            self.preflight_loopback_peer(request.url()).await?;
        }

        let response = self
            .inner
            .execute(request)
            .await
            .map_err(GatewayClientError::Request)?;
        self.read_capped(response).await
    }

    async fn preflight_loopback_peer(&self, url: &Url) -> Result<(), GatewayClientError> {
        let ip = loopback_ip_from_url(url)?;
        let port = url
            .port_or_known_default()
            .ok_or(GatewayClientError::MissingPort)?;
        let address = SocketAddr::new(ip, port);

        // reqwest does not expose its socket before writing the HTTP request. Auth-none
        // calls therefore pre-flight a TCP connection, inspect getpeername, close it,
        // then let reqwest connect to the same literal IP with redirects disabled.
        let stream = timeout(self.connect_timeout, TcpStream::connect(address))
            .await
            .map_err(|_| GatewayClientError::PeerConnectTimeout {
                timeout: self.connect_timeout,
            })?
            .map_err(|source| GatewayClientError::PeerConnect { address, source })?;
        let peer = stream
            .peer_addr()
            .map_err(GatewayClientError::PeerAddress)?;
        if !is_allowed_loopback_ip(peer.ip()) {
            return Err(GatewayClientError::PeerNotLoopback { peer });
        }
        drop(stream);
        Ok(())
    }

    async fn read_capped(
        &self,
        mut response: reqwest::Response,
    ) -> Result<CappedResponse, GatewayClientError> {
        let status = response.status();
        let headers = response.headers().clone();
        let mut body = Vec::with_capacity(
            response
                .content_length()
                .and_then(|length| usize::try_from(length).ok())
                .unwrap_or(0)
                .min(self.response_body_limit),
        );

        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(GatewayClientError::Request)?
        {
            let received = body.len().saturating_add(chunk.len());
            if received > self.response_body_limit {
                return Err(GatewayClientError::BodyTooLarge {
                    limit: self.response_body_limit,
                    received,
                });
            }
            body.extend_from_slice(&chunk);
        }

        Ok(CappedResponse {
            status,
            headers,
            body,
        })
    }
}

pub(super) fn parse_loopback_host(raw_host: &str) -> Result<IpAddr, GatewayClientError> {
    if raw_host.contains('%') {
        return Err(invalid_host(raw_host, "zone-qualified IPv6 is forbidden"));
    }

    let parsed = Host::parse(raw_host).or_else(|error| {
        if raw_host.contains(':') && !raw_host.starts_with('[') {
            Host::parse(&format!("[{raw_host}]"))
        } else {
            Err(error)
        }
    });
    match parsed {
        Ok(Host::Ipv4(ip)) if ip.is_loopback() && raw_host == ip.to_string() => Ok(IpAddr::V4(ip)),
        Ok(Host::Ipv4(ip)) if ip.is_loopback() => Err(invalid_host(
            raw_host,
            "IPv4 loopback must use canonical dotted-decimal notation",
        )),
        Ok(Host::Ipv4(_)) => Err(invalid_host(raw_host, "IPv4 address is not in 127.0.0.0/8")),
        Ok(Host::Ipv6(ip)) if ip == Ipv6Addr::LOCALHOST => Ok(IpAddr::V6(ip)),
        Ok(Host::Ipv6(_)) => Err(invalid_host(raw_host, "IPv6 address is not exactly ::1")),
        Ok(Host::Domain(_)) => Err(invalid_host(
            raw_host,
            "hostnames are forbidden for auth-none endpoints",
        )),
        Err(_) => Err(invalid_host(raw_host, "host syntax is invalid")),
    }
}

fn loopback_ip_from_url(url: &Url) -> Result<IpAddr, GatewayClientError> {
    match url.host().ok_or(GatewayClientError::MissingHost)? {
        Host::Ipv4(ip) if ip.is_loopback() => Ok(IpAddr::V4(ip)),
        Host::Ipv4(_) => Err(invalid_host(
            url.host_str().unwrap_or_default(),
            "IPv4 address is not in 127.0.0.0/8",
        )),
        Host::Ipv6(ip) if ip == Ipv6Addr::LOCALHOST => Ok(IpAddr::V6(ip)),
        Host::Ipv6(_) => Err(invalid_host(
            url.host_str().unwrap_or_default(),
            "IPv6 address is not exactly ::1",
        )),
        Host::Domain(host) => Err(invalid_host(
            host,
            "hostnames are forbidden for auth-none endpoints",
        )),
    }
}

fn is_allowed_loopback_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.is_loopback(),
        IpAddr::V6(ip) => ip == Ipv6Addr::LOCALHOST,
    }
}

fn invalid_host(host: &str, reason: &str) -> GatewayClientError {
    GatewayClientError::InvalidLoopbackHost {
        host: host.to_string(),
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::mock::{MockBehavior, MockProvider};

    #[test]
    fn loopback_host_parser_matrix() {
        let accepted = [
            ("127.0.0.1", "127.0.0.1"),
            ("127.0.0.255", "127.0.0.255"),
            ("127.42.18.9", "127.42.18.9"),
            ("127.255.255.255", "127.255.255.255"),
            ("::1", "::1"),
            ("[::1]", "::1"),
        ];
        for (raw, expected) in accepted {
            assert_eq!(
                parse_loopback_host(raw).unwrap().to_string(),
                expected,
                "{raw}"
            );
        }

        let rejected = [
            "localhost",
            "LOCALHOST",
            "loopback.local",
            "0.0.0.0",
            "126.255.255.255",
            "128.0.0.0",
            "127.1",
            "127.0.1",
            "2130706433",
            "0x7f000001",
            "::",
            "[::]",
            "::ffff:127.0.0.1",
            "[::ffff:127.0.0.1]",
            "::1%en0",
            "[::1%25en0]",
            "2001:db8::1",
            "",
        ];
        for raw in rejected {
            assert!(
                parse_loopback_host(raw).is_err(),
                "unexpectedly accepted {raw}"
            );
        }
    }

    #[tokio::test]
    async fn loopback_request_preflights_and_reads_a_capped_response() {
        let provider = MockProvider::start().await.unwrap();
        provider.enqueue("/embeddings", MockBehavior::Ok);
        let client =
            GatewayHttpClient::new(Duration::from_secs(1), Duration::from_secs(1), 16 * 1024)
                .unwrap();
        let request = client
            .request(Method::POST, provider.url("/embeddings"))
            .json(&serde_json::json!({"model":"m","input":["hello"],"dimensions":2}))
            .build()
            .unwrap();

        let response = client
            .execute(request, EndpointSecurity::LoopbackAuthNone)
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::OK);
        assert!(response.body.len() < 16 * 1024);
    }

    #[tokio::test]
    async fn auth_none_rejects_hostname_before_sending_request() {
        let client =
            GatewayHttpClient::new(Duration::from_millis(50), Duration::from_millis(50), 1024)
                .unwrap();
        let request = client
            .request(Method::GET, Url::parse("http://localhost:9/").unwrap())
            .build()
            .unwrap();

        let error = client
            .execute(request, EndpointSecurity::LoopbackAuthNone)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            GatewayClientError::InvalidLoopbackHost { .. }
        ));
    }

    #[tokio::test]
    async fn body_cap_is_enforced_while_streaming() {
        let provider = MockProvider::start().await.unwrap();
        provider.enqueue("/large", MockBehavior::OversizedBody { bytes: 4097 });
        let client =
            GatewayHttpClient::new(Duration::from_secs(1), Duration::from_secs(1), 4096).unwrap();
        let request = client
            .request(Method::GET, provider.url("/large"))
            .build()
            .unwrap();

        let error = client
            .execute(request, EndpointSecurity::LoopbackAuthNone)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            GatewayClientError::BodyTooLarge {
                limit: 4096,
                received: 4097
            }
        ));
    }

    #[tokio::test]
    async fn redirects_are_returned_instead_of_followed() {
        let provider = MockProvider::start().await.unwrap();
        provider.enqueue(
            "/redirect",
            MockBehavior::Redirect {
                location: "/target".to_string(),
            },
        );
        provider.enqueue("/target", MockBehavior::Ok);
        let client =
            GatewayHttpClient::new(Duration::from_secs(1), Duration::from_secs(1), 4096).unwrap();
        let request = client
            .request(Method::GET, provider.url("/redirect"))
            .build()
            .unwrap();

        let response = client
            .execute(request, EndpointSecurity::LoopbackAuthNone)
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::FOUND);
        assert_eq!(provider.requests_for("/target").len(), 0);
    }

    #[tokio::test]
    async fn caller_read_timeout_applies_to_hanging_responses() {
        let provider = MockProvider::start().await.unwrap();
        provider.enqueue(
            "/hang",
            MockBehavior::Hang {
                duration: Duration::from_millis(200),
            },
        );
        let client =
            GatewayHttpClient::new(Duration::from_secs(1), Duration::from_millis(25), 4096)
                .unwrap();
        let request = client
            .request(Method::GET, provider.url("/hang"))
            .build()
            .unwrap();

        let error = client
            .execute(request, EndpointSecurity::LoopbackAuthNone)
            .await
            .unwrap_err();
        assert!(matches!(error, GatewayClientError::Request(source) if source.is_timeout()));
    }
}
