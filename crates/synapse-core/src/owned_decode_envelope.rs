//! Shared wire contract for the `owned-decode-envelope-v2` stream.
//!
//! The contract deliberately keeps stream accounting separate from transport
//! delivery. A lost or gapped frame is reconciled through [`SessionStatus`],
//! while accepted progress and terminal frames never revise committed history.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::Fingerprint;

/// Stable schema identifier for the owned decode streaming extension.
pub const OWNED_DECODE_ENVELOPE_V2_SCHEMA: &str = "owned-decode-envelope-v2";
/// Wire protocol version carried by [`FrameEnvelope`].
pub const OWNED_DECODE_ENVELOPE_V2_PROTOCOL_VERSION: u8 = 2;

/// Generation modes accepted by the shared admission contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum GenerationMode {
    GreedyTop1,
    TopK { k: u32 },
    TopP { p: f64 },
    Temperature { temperature: f64 },
}

/// Generation configuration checked before dispatching a greedy-only owned decode request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfiguration {
    pub mode: GenerationMode,
}

/// Stable refusal vocabulary for owned decode admission and continuation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnedDecodeRefusal {
    ArtifactUnapproved,
    ArtifactMismatch,
    ArtifactDisabled,
    ArtifactRevoked,
    UnsupportedMachine,
    UnsupportedQuantization,
    SamplingUnsupported,
    InsufficientMemory,
    IncompatibleResidentArtifact,
    InvalidContextCeiling,
    InvalidKvConfiguration,
    InvalidKvAlignment,
    RetainedKvUnavailable,
    SessionStillInFlight,
    FailedSessionContinuation,
}

impl OwnedDecodeRefusal {
    /// Return the frozen wire literal for this refusal.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArtifactUnapproved => "artifact_unapproved",
            Self::ArtifactMismatch => "artifact_mismatch",
            Self::ArtifactDisabled => "artifact_disabled",
            Self::ArtifactRevoked => "artifact_revoked",
            Self::UnsupportedMachine => "unsupported_machine",
            Self::UnsupportedQuantization => "unsupported_quantization",
            Self::SamplingUnsupported => "owned_decode_sampling_unsupported",
            Self::InsufficientMemory => "insufficient_memory",
            Self::IncompatibleResidentArtifact => "incompatible_resident_artifact",
            Self::InvalidContextCeiling => "invalid_context_ceiling",
            Self::InvalidKvConfiguration => "invalid_kv_configuration",
            Self::InvalidKvAlignment => "invalid_kv_alignment",
            Self::RetainedKvUnavailable => "retained_kv_unavailable",
            Self::SessionStillInFlight => "session_still_in_flight",
            Self::FailedSessionContinuation => "failed_session_continuation",
        }
    }
}

/// Approval state of the artifact selected for a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactServingState {
    Approved,
    Disabled,
    Revoked,
}

/// Reject disabled and revoked artifacts before admission or continuation.
pub const fn validate_artifact_serving_state(
    state: ArtifactServingState,
) -> Result<(), OwnedDecodeRefusal> {
    match state {
        ArtifactServingState::Approved => Ok(()),
        ArtifactServingState::Disabled => Err(OwnedDecodeRefusal::ArtifactDisabled),
        ArtifactServingState::Revoked => Err(OwnedDecodeRefusal::ArtifactRevoked),
    }
}

/// Enforce the greedy-only owned decode policy before any token commits.
pub fn validate_greedy_generation(
    configuration: &GenerationConfiguration,
) -> Result<(), OwnedDecodeRefusal> {
    match &configuration.mode {
        GenerationMode::GreedyTop1 => Ok(()),
        GenerationMode::TopK { .. }
        | GenerationMode::TopP { .. }
        | GenerationMode::Temperature { .. } => Err(OwnedDecodeRefusal::SamplingUnsupported),
    }
}

/// A requested snapshot or reuse boundary for the per-session KV table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvReuseBoundary {
    pub position: u32,
    pub block_size: u32,
    pub recurrent_state_grain: u32,
}

/// Reject invalid KV configuration and boundaries that are not LCM-aligned.
pub fn validate_kv_reuse_boundary(boundary: KvReuseBoundary) -> Result<(), OwnedDecodeRefusal> {
    if !matches!(boundary.block_size, 256 | 512 | 1024) || boundary.recurrent_state_grain == 0 {
        return Err(OwnedDecodeRefusal::InvalidKvConfiguration);
    }

    let alignment = lcm(boundary.block_size, boundary.recurrent_state_grain)
        .ok_or(OwnedDecodeRefusal::InvalidKvConfiguration)?;
    if boundary.position % alignment != 0 {
        return Err(OwnedDecodeRefusal::InvalidKvAlignment);
    }

    Ok(())
}

fn lcm(left: u32, right: u32) -> Option<u32> {
    left.checked_div(gcd(left, right))?.checked_mul(right)
}

const fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// A per-request stream position. The first accepted frame is sequence one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StreamSequence(pub u64);

impl StreamSequence {
    /// The first sequence number emitted for a request.
    pub const FIRST: Self = Self(1);

    fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

/// Newly committed tokens and their cumulative count.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgressFrame {
    pub committed_token_ids: Vec<u32>,
    pub committed_token_count: u32,
}

/// Whether the terminal response was decoded serially or speculatively.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeMode {
    Serial,
    Speculative,
}

/// Required telemetry for a speculative terminal response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeTelemetry {
    pub proposed_depth: u32,
    pub accepted_depth: u32,
    pub acceptance_rate: f64,
    pub verification_work: u64,
    pub controller_decisions: Vec<String>,
}

/// Terminal state that never changes previously committed token history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalState {
    Completed,
    Aborted,
    ArtifactDisabled,
    ArtifactRevoked,
    Failed,
}

/// Identity shared by a terminal stream frame and its corresponding oneshot.
///
/// The values are opaque here because their hashes are owned by the respective
/// decode, processing, runtime, and worker-generation identity calculators.
/// The contract preserves them exactly: a decode identity rotation changes
/// `decode_fingerprint`; processing and runtime rotations change their own
/// fields; and a worker restart changes `worker_generation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OneshotEnvelopeIdentity {
    pub decode_fingerprint: Fingerprint,
    pub processing_fingerprint: Fingerprint,
    pub runtime_config_digest: String,
    pub worker_generation: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_digest: Option<String>,
}

/// Required terminal accounting and the complete oneshot response identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalEnvelope {
    pub req_id: String,
    pub session_id: String,
    pub committed_token_count: u32,
    pub tokens_emitted: u32,
    #[serde(flatten)]
    pub identity: OneshotEnvelopeIdentity,
    pub terminal_state: TerminalState,
    pub decode_mode: DecodeMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_telemetry: Option<SpeculativeTelemetry>,
}

/// The ordered worker-frame vocabulary for the version-two stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerFrame {
    Progress { progress: ProgressFrame },
    Final { terminal: TerminalEnvelope },
    Error { terminal: TerminalEnvelope },
}

/// A versioned worker frame correlated to exactly one decode request and session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameEnvelope {
    pub protocol: String,
    pub protocol_version: u8,
    pub req_id: String,
    pub session_id: String,
    pub stream_seq: StreamSequence,
    #[serde(flatten)]
    pub frame: WorkerFrame,
}

impl FrameEnvelope {
    /// Build a version-two envelope with the fixed schema and protocol version.
    #[must_use]
    pub fn new(
        req_id: impl Into<String>,
        session_id: impl Into<String>,
        stream_seq: StreamSequence,
        frame: WorkerFrame,
    ) -> Self {
        Self {
            protocol: OWNED_DECODE_ENVELOPE_V2_SCHEMA.to_string(),
            protocol_version: OWNED_DECODE_ENVELOPE_V2_PROTOCOL_VERSION,
            req_id: req_id.into(),
            session_id: session_id.into(),
            stream_seq,
            frame,
        }
    }
}

/// Authoritative request state returned after a stream gap or lost terminal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "terminal_state", rename_all = "snake_case")]
pub enum SessionStatusState {
    InFlight,
    Terminal(TerminalState),
}

/// Status recovery response for a request within an admitted session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionStatus {
    pub session_id: String,
    pub req_id: String,
    pub committed_token_count: u32,
    pub state: SessionStatusState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_kv_session_id: Option<String>,
}

/// Validate a continuation request after status recovery.
///
/// A stream cannot be resumed while it is in flight. A new continuation requires
/// a retained KV session and is refused permanently when committed history could
/// not be established and the session entered the failed terminal state.
pub fn validate_continuation(
    artifact_state: ArtifactServingState,
    status: &SessionStatus,
) -> Result<(), OwnedDecodeRefusal> {
    validate_artifact_serving_state(artifact_state)?;

    match status.state {
        SessionStatusState::InFlight => Err(OwnedDecodeRefusal::SessionStillInFlight),
        SessionStatusState::Terminal(TerminalState::Failed) => {
            Err(OwnedDecodeRefusal::FailedSessionContinuation)
        }
        SessionStatusState::Terminal(TerminalState::ArtifactDisabled) => {
            Err(OwnedDecodeRefusal::ArtifactDisabled)
        }
        SessionStatusState::Terminal(TerminalState::ArtifactRevoked) => {
            Err(OwnedDecodeRefusal::ArtifactRevoked)
        }
        SessionStatusState::Terminal(TerminalState::Completed | TerminalState::Aborted) => {
            if status
                .retained_kv_session_id
                .as_deref()
                .is_some_and(|session_id| !session_id.is_empty())
            {
                Ok(())
            } else {
                Err(OwnedDecodeRefusal::RetainedKvUnavailable)
            }
        }
    }
}

/// The identity and accounting fields of `DecodeTransportResponse::Cancelled`.
///
/// The transport enum remains free to carry other responses, but an abort
/// acknowledgement must retain this exact three-field shape for status recovery.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancelledTransportResponse {
    pub req_id: String,
    pub generation_id: String,
    pub committed_token_count: u32,
}

impl CancelledTransportResponse {
    /// Check that abort acknowledgement agrees with the authoritative status.
    pub fn validate_against_status(
        &self,
        status: &SessionStatus,
    ) -> Result<(), EnvelopeValidationError> {
        if self.req_id != status.req_id {
            return Err(EnvelopeValidationError::RequestIdMismatch);
        }
        if self.committed_token_count != status.committed_token_count {
            return Err(EnvelopeValidationError::StatusAccountingMismatch {
                expected: status.committed_token_count,
                actual: self.committed_token_count,
            });
        }
        Ok(())
    }
}

/// Result of observing a frame in a request stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamFrameDisposition {
    Accepted,
    Duplicate,
    Gap {
        expected: StreamSequence,
        received: StreamSequence,
    },
}

/// Stateful validator for one request's ordered, committed token stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamContractValidator {
    req_id: String,
    session_id: String,
    last_stream_seq: Option<StreamSequence>,
    committed_token_count: u32,
    terminal_observed: bool,
}

impl StreamContractValidator {
    /// Start validating frames for one request and admitted session.
    #[must_use]
    pub fn new(req_id: impl Into<String>, session_id: impl Into<String>) -> Self {
        Self {
            req_id: req_id.into(),
            session_id: session_id.into(),
            last_stream_seq: None,
            committed_token_count: 0,
            terminal_observed: false,
        }
    }

    /// Return the committed-token count accepted so far.
    #[must_use]
    pub const fn committed_token_count(&self) -> u32 {
        self.committed_token_count
    }

    /// Observe one frame. Duplicates are idempotent; gaps require status recovery.
    pub fn observe(
        &mut self,
        envelope: &FrameEnvelope,
    ) -> Result<StreamFrameDisposition, EnvelopeValidationError> {
        self.validate_header(envelope)?;

        if envelope.stream_seq.0 == 0 {
            return Err(EnvelopeValidationError::ZeroStreamSequence);
        }

        let expected = match self.last_stream_seq {
            Some(last_stream_seq) => last_stream_seq
                .next()
                .ok_or(EnvelopeValidationError::StreamSequenceOverflow)?,
            None => StreamSequence::FIRST,
        };
        if let Some(last_stream_seq) = self.last_stream_seq {
            if envelope.stream_seq <= last_stream_seq {
                return Ok(StreamFrameDisposition::Duplicate);
            }
        }
        if envelope.stream_seq != expected {
            return Ok(StreamFrameDisposition::Gap {
                expected,
                received: envelope.stream_seq,
            });
        }
        if self.terminal_observed {
            return Err(EnvelopeValidationError::FrameAfterTerminal);
        }

        match &envelope.frame {
            WorkerFrame::Progress { progress } => self.validate_progress(progress)?,
            WorkerFrame::Final { terminal } => {
                self.validate_terminal(envelope, terminal, TerminalFrameKind::Final)?
            }
            WorkerFrame::Error { terminal } => {
                self.validate_terminal(envelope, terminal, TerminalFrameKind::Error)?
            }
        }
        self.last_stream_seq = Some(envelope.stream_seq);
        Ok(StreamFrameDisposition::Accepted)
    }

    fn validate_header(&self, envelope: &FrameEnvelope) -> Result<(), EnvelopeValidationError> {
        if envelope.protocol != OWNED_DECODE_ENVELOPE_V2_SCHEMA {
            return Err(EnvelopeValidationError::ProtocolMismatch);
        }
        if envelope.protocol_version != OWNED_DECODE_ENVELOPE_V2_PROTOCOL_VERSION {
            return Err(EnvelopeValidationError::ProtocolVersionMismatch {
                actual: envelope.protocol_version,
            });
        }
        if envelope.req_id.is_empty() || envelope.session_id.is_empty() {
            return Err(EnvelopeValidationError::MissingFrameCorrelation);
        }
        if envelope.req_id != self.req_id {
            return Err(EnvelopeValidationError::RequestIdMismatch);
        }
        if envelope.session_id != self.session_id {
            return Err(EnvelopeValidationError::SessionIdMismatch);
        }
        Ok(())
    }

    fn validate_progress(
        &mut self,
        progress: &ProgressFrame,
    ) -> Result<(), EnvelopeValidationError> {
        let added = u32::try_from(progress.committed_token_ids.len())
            .map_err(|_| EnvelopeValidationError::TokenCountOverflow)?;
        let expected = self
            .committed_token_count
            .checked_add(added)
            .ok_or(EnvelopeValidationError::TokenCountOverflow)?;
        if progress.committed_token_count != expected {
            return Err(EnvelopeValidationError::ProgressAccountingMismatch {
                expected,
                actual: progress.committed_token_count,
            });
        }
        self.committed_token_count = expected;
        Ok(())
    }

    fn validate_terminal(
        &mut self,
        envelope: &FrameEnvelope,
        terminal: &TerminalEnvelope,
        frame_kind: TerminalFrameKind,
    ) -> Result<(), EnvelopeValidationError> {
        if terminal.req_id != envelope.req_id {
            return Err(EnvelopeValidationError::RequestIdMismatch);
        }
        if terminal.session_id != envelope.session_id {
            return Err(EnvelopeValidationError::SessionIdMismatch);
        }
        if terminal.tokens_emitted != terminal.committed_token_count {
            return Err(EnvelopeValidationError::TerminalAccountingMismatch {
                committed_token_count: terminal.committed_token_count,
                tokens_emitted: terminal.tokens_emitted,
            });
        }
        if terminal.committed_token_count != self.committed_token_count {
            return Err(EnvelopeValidationError::TerminalCommittedCountMismatch {
                expected: self.committed_token_count,
                actual: terminal.committed_token_count,
            });
        }
        match (frame_kind, terminal.terminal_state) {
            (TerminalFrameKind::Final, TerminalState::Completed)
            | (TerminalFrameKind::Error, TerminalState::Aborted)
            | (TerminalFrameKind::Error, TerminalState::ArtifactDisabled)
            | (TerminalFrameKind::Error, TerminalState::ArtifactRevoked)
            | (TerminalFrameKind::Error, TerminalState::Failed) => {}
            _ => return Err(EnvelopeValidationError::TerminalFrameStateMismatch),
        }
        validate_terminal_identity(&terminal.identity)?;
        match (&terminal.decode_mode, &terminal.speculative_telemetry) {
            (DecodeMode::Speculative, None) => {
                return Err(EnvelopeValidationError::MissingSpeculativeTelemetry)
            }
            (DecodeMode::Serial, Some(_)) => {
                return Err(EnvelopeValidationError::UnexpectedSpeculativeTelemetry)
            }
            (DecodeMode::Speculative, Some(telemetry))
                if !telemetry.acceptance_rate.is_finite()
                    || !(0.0..=1.0).contains(&telemetry.acceptance_rate) =>
            {
                return Err(EnvelopeValidationError::InvalidAcceptanceRate)
            }
            _ => {}
        }
        self.terminal_observed = true;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum TerminalFrameKind {
    Final,
    Error,
}

fn validate_terminal_identity(
    identity: &OneshotEnvelopeIdentity,
) -> Result<(), EnvelopeValidationError> {
    if identity.decode_fingerprint.0.is_empty() {
        return Err(EnvelopeValidationError::MissingIdentityField(
            "decode_fingerprint",
        ));
    }
    if identity.processing_fingerprint.0.is_empty() {
        return Err(EnvelopeValidationError::MissingIdentityField(
            "processing_fingerprint",
        ));
    }
    if identity.runtime_config_digest.is_empty() {
        return Err(EnvelopeValidationError::MissingIdentityField(
            "runtime_config_digest",
        ));
    }
    if identity
        .derived_digest
        .as_deref()
        .is_some_and(|digest| digest.is_empty())
    {
        return Err(EnvelopeValidationError::MissingIdentityField(
            "derived_digest",
        ));
    }
    Ok(())
}

/// Structural or accounting violation of the version-two envelope contract.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum EnvelopeValidationError {
    #[error("frame uses a different envelope schema")]
    ProtocolMismatch,
    #[error("frame uses unsupported protocol version {actual}")]
    ProtocolVersionMismatch { actual: u8 },
    #[error("frame correlation requires non-empty request and session IDs")]
    MissingFrameCorrelation,
    #[error("frame request ID does not match the request stream")]
    RequestIdMismatch,
    #[error("frame session ID does not match the admitted session")]
    SessionIdMismatch,
    #[error("stream sequence zero is invalid")]
    ZeroStreamSequence,
    #[error("stream sequence cannot advance past u64::MAX")]
    StreamSequenceOverflow,
    #[error("token count cannot fit in u32")]
    TokenCountOverflow,
    #[error("progress committed-token count {actual} does not equal {expected}")]
    ProgressAccountingMismatch { expected: u32, actual: u32 },
    #[error(
        "terminal tokens emitted {tokens_emitted} does not equal committed-token count {committed_token_count}"
    )]
    TerminalAccountingMismatch {
        committed_token_count: u32,
        tokens_emitted: u32,
    },
    #[error("terminal committed-token count {actual} does not equal stream count {expected}")]
    TerminalCommittedCountMismatch { expected: u32, actual: u32 },
    #[error("terminal frame kind does not match its terminal state")]
    TerminalFrameStateMismatch,
    #[error("received a new frame after the terminal frame")]
    FrameAfterTerminal,
    #[error("speculative terminal response is missing telemetry")]
    MissingSpeculativeTelemetry,
    #[error("serial terminal response must not carry speculative telemetry")]
    UnexpectedSpeculativeTelemetry,
    #[error("speculative acceptance rate must be finite and in [0, 1]")]
    InvalidAcceptanceRate,
    #[error("terminal identity field {0} is required")]
    MissingIdentityField(&'static str),
    #[error("abort acknowledgement count {actual} does not equal status count {expected}")]
    StatusAccountingMismatch { expected: u32, actual: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> OneshotEnvelopeIdentity {
        OneshotEnvelopeIdentity {
            decode_fingerprint: Fingerprint("decode-fingerprint".to_string()),
            processing_fingerprint: Fingerprint("processing-fingerprint".to_string()),
            runtime_config_digest: "runtime-digest".to_string(),
            worker_generation: 7,
            derived_digest: Some("derived-digest".to_string()),
        }
    }

    fn progress(sequence: u64, token_ids: Vec<u32>, committed_token_count: u32) -> FrameEnvelope {
        FrameEnvelope::new(
            "request-1",
            "session-1",
            StreamSequence(sequence),
            WorkerFrame::Progress {
                progress: ProgressFrame {
                    committed_token_ids: token_ids,
                    committed_token_count,
                },
            },
        )
    }

    fn terminal(
        sequence: u64,
        frame: fn(TerminalEnvelope) -> WorkerFrame,
        terminal_state: TerminalState,
        committed_token_count: u32,
    ) -> FrameEnvelope {
        FrameEnvelope::new(
            "request-1",
            "session-1",
            StreamSequence(sequence),
            frame(TerminalEnvelope {
                req_id: "request-1".to_string(),
                session_id: "session-1".to_string(),
                committed_token_count,
                tokens_emitted: committed_token_count,
                identity: identity(),
                terminal_state,
                decode_mode: DecodeMode::Serial,
                speculative_telemetry: None,
            }),
        )
    }

    fn final_frame(terminal: TerminalEnvelope) -> WorkerFrame {
        WorkerFrame::Final { terminal }
    }

    fn error_frame(terminal: TerminalEnvelope) -> WorkerFrame {
        WorkerFrame::Error { terminal }
    }

    #[test]
    fn v2_envelopes_are_versioned_and_correlate_every_frame() {
        let frame = progress(1, vec![41], 1);
        let value = serde_json::to_value(&frame).expect("serialize frame");
        assert_eq!(value["protocol"], OWNED_DECODE_ENVELOPE_V2_SCHEMA);
        assert_eq!(value["protocol_version"], 2);
        assert_eq!(value["req_id"], "request-1");
        assert_eq!(value["stream_seq"], 1);
        assert!(serde_json::from_value::<FrameEnvelope>(serde_json::json!({
            "protocol": OWNED_DECODE_ENVELOPE_V2_SCHEMA,
            "protocol_version": 2,
            "req_id": "request-1",
            "session_id": "session-1",
            "stream_seq": 1,
            "kind": "progress",
            "progress": { "committed_token_ids": [], "committed_token_count": 0 },
            "unknown": true
        }))
        .is_err());
    }

    #[test]
    fn stream_validation_is_monotonic_idempotent_and_terminal_accounted() {
        let mut validator = StreamContractValidator::new("request-1", "session-1");
        let first = progress(1, vec![10, 11], 2);
        assert_eq!(
            validator.observe(&first),
            Ok(StreamFrameDisposition::Accepted)
        );
        assert_eq!(validator.committed_token_count(), 2);
        assert_eq!(
            validator.observe(&first),
            Ok(StreamFrameDisposition::Duplicate),
            "duplicate sequence numbers are idempotent"
        );
        assert_eq!(
            validator.observe(&progress(3, vec![12], 3)),
            Ok(StreamFrameDisposition::Gap {
                expected: StreamSequence(2),
                received: StreamSequence(3),
            })
        );
        assert_eq!(
            validator.observe(&progress(2, vec![12], 3)),
            Ok(StreamFrameDisposition::Accepted)
        );
        assert_eq!(
            validator.observe(&terminal(3, final_frame, TerminalState::Completed, 3)),
            Ok(StreamFrameDisposition::Accepted)
        );
    }

    #[test]
    fn terminal_rejects_identity_or_accounting_drift() {
        let mut validator = StreamContractValidator::new("request-1", "session-1");
        assert_eq!(
            validator.observe(&progress(1, vec![10], 1)),
            Ok(StreamFrameDisposition::Accepted)
        );

        let mut mismatched = terminal(2, final_frame, TerminalState::Completed, 1);
        let WorkerFrame::Final {
            terminal: terminal_response,
        } = &mut mismatched.frame
        else {
            unreachable!("fixture builds a final frame")
        };
        terminal_response.tokens_emitted = 2;
        assert_eq!(
            validator.observe(&mismatched),
            Err(EnvelopeValidationError::TerminalAccountingMismatch {
                committed_token_count: 1,
                tokens_emitted: 2,
            })
        );

        let mut missing_identity = terminal(2, final_frame, TerminalState::Completed, 1);
        let WorkerFrame::Final {
            terminal: terminal_response,
        } = &mut missing_identity.frame
        else {
            unreachable!("fixture builds a final frame")
        };
        terminal_response.identity.processing_fingerprint = Fingerprint(String::new());
        assert_eq!(
            validator.observe(&missing_identity),
            Err(EnvelopeValidationError::MissingIdentityField(
                "processing_fingerprint"
            ))
        );
    }

    #[test]
    fn terminal_frame_kind_and_speculative_telemetry_are_checked() {
        let mut validator = StreamContractValidator::new("request-1", "session-1");
        let mut missing_telemetry = terminal(1, final_frame, TerminalState::Completed, 0);
        let WorkerFrame::Final {
            terminal: terminal_response,
        } = &mut missing_telemetry.frame
        else {
            unreachable!("fixture builds a final frame")
        };
        terminal_response.decode_mode = DecodeMode::Speculative;
        assert_eq!(
            validator.observe(&missing_telemetry),
            Err(EnvelopeValidationError::MissingSpeculativeTelemetry)
        );

        let mut validator = StreamContractValidator::new("request-1", "session-1");
        assert_eq!(
            validator.observe(&terminal(1, error_frame, TerminalState::Completed, 0)),
            Err(EnvelopeValidationError::TerminalFrameStateMismatch)
        );
    }

    #[test]
    fn shared_refusal_validators_fail_closed_with_stable_literals() {
        assert_eq!(
            validate_greedy_generation(&GenerationConfiguration {
                mode: GenerationMode::Temperature { temperature: 0.8 },
            }),
            Err(OwnedDecodeRefusal::SamplingUnsupported)
        );
        assert_eq!(
            OwnedDecodeRefusal::SamplingUnsupported.as_str(),
            "owned_decode_sampling_unsupported"
        );
        assert_eq!(
            validate_artifact_serving_state(ArtifactServingState::Disabled),
            Err(OwnedDecodeRefusal::ArtifactDisabled)
        );
        assert_eq!(
            validate_artifact_serving_state(ArtifactServingState::Revoked),
            Err(OwnedDecodeRefusal::ArtifactRevoked)
        );
        assert_eq!(
            validate_kv_reuse_boundary(KvReuseBoundary {
                position: 513,
                block_size: 256,
                recurrent_state_grain: 128,
            }),
            Err(OwnedDecodeRefusal::InvalidKvAlignment)
        );
        assert_eq!(
            validate_kv_reuse_boundary(KvReuseBoundary {
                position: 512,
                block_size: 128,
                recurrent_state_grain: 128,
            }),
            Err(OwnedDecodeRefusal::InvalidKvConfiguration)
        );
    }

    #[test]
    fn failed_session_refuses_continuation_and_abort_shape_agrees_with_status() {
        let failed = SessionStatus {
            session_id: "session-1".to_string(),
            req_id: "request-1".to_string(),
            committed_token_count: 5,
            state: SessionStatusState::Terminal(TerminalState::Failed),
            retained_kv_session_id: Some("retained-1".to_string()),
        };
        assert_eq!(
            validate_continuation(ArtifactServingState::Approved, &failed),
            Err(OwnedDecodeRefusal::FailedSessionContinuation)
        );

        let cancelled = CancelledTransportResponse {
            req_id: "request-1".to_string(),
            generation_id: "generation-1".to_string(),
            committed_token_count: 5,
        };
        assert_eq!(cancelled.validate_against_status(&failed), Ok(()));
        assert_eq!(
            serde_json::to_value(cancelled).expect("serialize abort response"),
            serde_json::json!({
                "req_id": "request-1",
                "generation_id": "generation-1",
                "committed_token_count": 5,
            })
        );
    }
}
