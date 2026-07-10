use std::time::{Duration, SystemTime};

use reqwest::StatusCode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ClassifierPreset {
    pub name: &'static str,
    transient_client_statuses: &'static [StatusCode],
}

pub(super) const CLASSIFIER_PRESETS: &[ClassifierPreset] = &[
    ClassifierPreset {
        name: "generic",
        transient_client_statuses: &[],
    },
    ClassifierPreset {
        name: "lmstudio",
        transient_client_statuses: &[StatusCode::BAD_REQUEST],
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProviderFailure<'a> {
    Http {
        status: StatusCode,
        retry_after: Option<&'a str>,
    },
    Connect,
    Timeout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FailureClass {
    Permanent,
    Transient,
    Pacing { retry_after: Option<Duration> },
}

pub(super) fn preset(name: &str) -> Option<&'static ClassifierPreset> {
    CLASSIFIER_PRESETS
        .iter()
        .find(|candidate| candidate.name == name)
}

pub(super) fn classify(
    preset: &ClassifierPreset,
    failure: ProviderFailure<'_>,
    now: SystemTime,
) -> FailureClass {
    match failure {
        ProviderFailure::Connect | ProviderFailure::Timeout => FailureClass::Transient,
        ProviderFailure::Http {
            status: StatusCode::TOO_MANY_REQUESTS,
            retry_after,
        } => FailureClass::Pacing {
            retry_after: retry_after.and_then(|value| parse_retry_after(value, now)),
        },
        ProviderFailure::Http { status, .. }
            if status.is_server_error() || preset.transient_client_statuses.contains(&status) =>
        {
            FailureClass::Transient
        }
        ProviderFailure::Http { .. } => FailureClass::Permanent,
    }
}

fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    httpdate::parse_http_date(value)
        .ok()
        .map(|deadline| deadline.duration_since(now).unwrap_or(Duration::ZERO))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn http(status: StatusCode) -> ProviderFailure<'static> {
        ProviderFailure::Http {
            status,
            retry_after: None,
        }
    }

    #[test]
    fn preset_table_contains_only_the_frozen_day_one_rows() {
        assert_eq!(
            CLASSIFIER_PRESETS
                .iter()
                .map(|preset| preset.name)
                .collect::<Vec<_>>(),
            vec!["generic", "lmstudio"]
        );
        assert!(preset("missing").is_none());
    }

    #[test]
    fn generic_preset_classifies_each_status_family_row() {
        let generic = preset("generic").unwrap();
        let now = SystemTime::UNIX_EPOCH;
        let cases = [
            (StatusCode::BAD_REQUEST, FailureClass::Permanent),
            (StatusCode::UNAUTHORIZED, FailureClass::Permanent),
            (StatusCode::REQUEST_TIMEOUT, FailureClass::Permanent),
            (StatusCode::NOT_FOUND, FailureClass::Permanent),
            (StatusCode::INTERNAL_SERVER_ERROR, FailureClass::Transient),
            (StatusCode::BAD_GATEWAY, FailureClass::Transient),
            (StatusCode::SERVICE_UNAVAILABLE, FailureClass::Transient),
        ];

        for (status, expected) in cases {
            assert_eq!(classify(generic, http(status), now), expected, "{status}");
        }
    }

    #[test]
    fn lmstudio_only_overrides_bad_request_as_transient_bounded_input() {
        let lmstudio = preset("lmstudio").unwrap();
        let now = SystemTime::UNIX_EPOCH;
        let cases = [
            (StatusCode::BAD_REQUEST, FailureClass::Transient),
            (StatusCode::UNAUTHORIZED, FailureClass::Permanent),
            (StatusCode::REQUEST_TIMEOUT, FailureClass::Permanent),
            (StatusCode::INTERNAL_SERVER_ERROR, FailureClass::Transient),
        ];

        for (status, expected) in cases {
            assert_eq!(classify(lmstudio, http(status), now), expected, "{status}");
        }
    }

    #[test]
    fn pacing_is_distinct_for_every_preset_and_parses_delta_seconds() {
        let now = SystemTime::UNIX_EPOCH;
        for preset in CLASSIFIER_PRESETS {
            assert_eq!(
                classify(
                    preset,
                    ProviderFailure::Http {
                        status: StatusCode::TOO_MANY_REQUESTS,
                        retry_after: Some(" 17 "),
                    },
                    now,
                ),
                FailureClass::Pacing {
                    retry_after: Some(Duration::from_secs(17))
                }
            );
        }
    }

    #[test]
    fn retry_after_http_date_is_relative_to_the_call_time() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let deadline = now + Duration::from_secs(31);
        let retry_after = httpdate::fmt_http_date(deadline);

        assert_eq!(
            classify(
                preset("generic").unwrap(),
                ProviderFailure::Http {
                    status: StatusCode::TOO_MANY_REQUESTS,
                    retry_after: Some(&retry_after),
                },
                now,
            ),
            FailureClass::Pacing {
                retry_after: Some(Duration::from_secs(31))
            }
        );
    }

    #[test]
    fn retry_after_past_dates_clamp_to_zero_and_invalid_values_are_absent() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let past = httpdate::fmt_http_date(now - Duration::from_secs(1));
        let generic = preset("generic").unwrap();

        for (value, expected) in [
            (past.as_str(), Some(Duration::ZERO)),
            ("not-a-date", None),
            ("-1", None),
        ] {
            assert_eq!(
                classify(
                    generic,
                    ProviderFailure::Http {
                        status: StatusCode::TOO_MANY_REQUESTS,
                        retry_after: Some(value),
                    },
                    now,
                ),
                FailureClass::Pacing {
                    retry_after: expected
                }
            );
        }
    }

    #[test]
    fn connect_and_timeout_failures_are_transient_for_every_preset() {
        for preset in CLASSIFIER_PRESETS {
            assert_eq!(
                classify(preset, ProviderFailure::Connect, SystemTime::UNIX_EPOCH),
                FailureClass::Transient
            );
            assert_eq!(
                classify(preset, ProviderFailure::Timeout, SystemTime::UNIX_EPOCH),
                FailureClass::Transient
            );
        }
    }
}
