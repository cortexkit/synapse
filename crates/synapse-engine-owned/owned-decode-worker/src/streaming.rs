//! Supervisor-owned state for `owned-decode-envelope-v2` streams.
//!
//! Worker frames are transport deliveries, not the source of truth. This state
//! machine records the committed boundary before a frame is exposed to a client,
//! so `session_status` remains authoritative after a client observes a sequence
//! gap or loses a terminal frame.

use std::collections::BTreeMap;

use synapse_core::{
    validate_continuation, ArtifactServingState, CancelledTransportResponse, DecodeMode,
    EnvelopeValidationError, FrameEnvelope as StreamFrameEnvelope, OneshotEnvelopeIdentity,
    OwnedDecodeRefusal, SessionStatus, SessionStatusState, SpeculativeTelemetry,
    StreamContractValidator, StreamFrameDisposition, StreamSequence, TerminalEnvelope,
    TerminalState, WorkerFrame as StreamWorkerFrame,
};
use thiserror::Error;

/// Immutable request data the supervisor binds to one worker stream.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamRequest {
    /// Correlates every frame and status lookup for this decode request.
    pub req_id: String,
    /// The admitted session that owns the request's resident state.
    pub session_id: String,
    /// The existing generation identity used by cancellation acknowledgements.
    pub generation_id: String,
    /// The complete identity that every terminal frame must preserve.
    pub identity: OneshotEnvelopeIdentity,
    /// The decode mode used to validate terminal telemetry.
    pub decode_mode: DecodeMode,
    /// Grammar-constrained streams use a single-chain configuration.
    pub grammar_constrained: bool,
    /// Effective chain span for this request.
    pub chain_k: u32,
}

/// Preflight result for abort-time KV retention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetentionPreflight {
    /// The caller did not ask to retain the resident KV state.
    NotRequested,
    /// Retention was requested but cannot be guaranteed at the committed boundary.
    Refused,
    /// A retained session is ready at exactly this committed-token position.
    Ready {
        retained_kv_session_id: String,
        retained_position: u32,
    },
}

/// Identity of KV retained after a successful abort.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedPrefix {
    /// The retained KV session used by a new continuation request.
    pub retained_kv_session_id: String,
    /// The authoritative prefix length represented by that retained state.
    pub retained_position: u32,
}

/// Result of an abort acknowledged at the current committed boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct AbortOutcome {
    /// The cancellation response returned to the transport layer for this abort.
    pub cancellation: CancelledTransportResponse,
    /// The stream's single error terminal for the abort.
    pub terminal: StreamFrameEnvelope,
    /// Present only when preflight proved retention at the committed boundary.
    pub retained_prefix: Option<RetainedPrefix>,
}

/// Result of recording a worker process death.
#[derive(Clone, Debug, PartialEq)]
pub enum WorkerDeathOutcome {
    /// The ordered history was complete, so the supervisor emitted terminal
    /// accounting with the committed count it can prove.
    /// Boxed: the envelope dwarfs the other variant (~312 vs ~80 bytes), and
    /// this outcome is constructed once per worker death, so the indirection
    /// costs nothing on any hot path.
    Terminal(Box<StreamFrameEnvelope>),
    /// A gap left committed history unprovable. The terminal status refuses every
    /// continuation instead of fabricating a stream terminal.
    FailedWithoutTerminal(SessionStatus),
}

/// Cleanup performed by one supervision cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SupervisionCycle {
    /// Number of non-retained terminal resources reclaimed in this cycle.
    pub reclaimed_requests: usize,
}

/// Failure to create, advance, recover, or continue a supervised stream.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum StreamingSupervisorError {
    #[error("a stream request requires non-empty request, session, and generation identities")]
    MissingRequestIdentity,
    #[error("grammar-constrained streams require chain_k=1")]
    GrammarChainKMustBeOne,
    #[error("the request is already registered for this session")]
    DuplicateRequest,
    #[error("the request is not registered for this session")]
    UnknownRequest,
    #[error("the request already has a terminal state")]
    AlreadyTerminal,
    #[error("the terminal identity differs from the request's oneshot identity")]
    TerminalIdentityMismatch,
    #[error("the terminal decode mode differs from the request decode mode")]
    TerminalDecodeModeMismatch,
    #[error("the next stream sequence cannot be represented")]
    StreamSequenceExhausted,
    #[error("retention preflight did not produce a non-empty retained session ID")]
    EmptyRetainedSessionId,
    #[error("retention preflight position {actual} differs from committed boundary {expected}")]
    RetentionPositionMismatch { expected: u32, actual: u32 },
    #[error("the retained session ID does not match the aborted request")]
    RetainedSessionIdMismatch,
    #[error("retained prefix position differs from the authoritative committed count")]
    RetainedPrefixAccountingMismatch,
    #[error("continuation refused: {0:?}")]
    ContinuationRefused(OwnedDecodeRefusal),
    #[error(transparent)]
    Envelope(#[from] EnvelopeValidationError),
}

struct RequestStream {
    request: StreamRequest,
    validator: StreamContractValidator,
    last_accepted_sequence: Option<StreamSequence>,
    terminal_state: Option<TerminalState>,
    retained_prefix: Option<RetainedPrefix>,
    cleanup_pending: bool,
    history_provable: bool,
}

impl RequestStream {
    fn new(request: StreamRequest) -> Self {
        Self {
            validator: StreamContractValidator::new(&request.req_id, &request.session_id),
            request,
            last_accepted_sequence: None,
            terminal_state: None,
            retained_prefix: None,
            cleanup_pending: false,
            history_provable: true,
        }
    }

    fn next_sequence(&self) -> Result<StreamSequence, StreamingSupervisorError> {
        match self.last_accepted_sequence {
            Some(sequence) => sequence
                .0
                .checked_add(1)
                .map(StreamSequence)
                .ok_or(StreamingSupervisorError::StreamSequenceExhausted),
            None => Ok(StreamSequence::FIRST),
        }
    }

    fn status(&self) -> SessionStatus {
        let state = match self.terminal_state {
            Some(terminal_state) => SessionStatusState::Terminal(terminal_state),
            None => SessionStatusState::InFlight,
        };
        SessionStatus {
            session_id: self.request.session_id.clone(),
            req_id: self.request.req_id.clone(),
            committed_token_count: self.validator.committed_token_count(),
            state,
            retained_kv_session_id: self
                .retained_prefix
                .as_ref()
                .map(|prefix| prefix.retained_kv_session_id.clone()),
        }
    }

    fn validate_expected_terminal(
        &self,
        frame: &StreamFrameEnvelope,
    ) -> Result<(), StreamingSupervisorError> {
        let terminal = match &frame.frame {
            StreamWorkerFrame::Progress { .. } => return Ok(()),
            StreamWorkerFrame::Final { terminal } | StreamWorkerFrame::Error { terminal } => {
                terminal
            }
        };
        if terminal.identity != self.request.identity {
            return Err(StreamingSupervisorError::TerminalIdentityMismatch);
        }
        if terminal.decode_mode != self.request.decode_mode {
            return Err(StreamingSupervisorError::TerminalDecodeModeMismatch);
        }
        Ok(())
    }
}

/// Supervisor authority for all active and terminal version-two streams.
///
/// Status records intentionally survive resource cleanup. They are the recovery
/// authority for a client that lost frames, while only non-retained resident
/// resources are reclaimed during the next supervision cycle.
#[derive(Default)]
pub struct StreamingSupervisor {
    requests: BTreeMap<(String, String), RequestStream>,
}

impl StreamingSupervisor {
    /// Register one admitted decode request before forwarding its first frame.
    pub fn begin(&mut self, request: StreamRequest) -> Result<(), StreamingSupervisorError> {
        if request.req_id.is_empty()
            || request.session_id.is_empty()
            || request.generation_id.is_empty()
        {
            return Err(StreamingSupervisorError::MissingRequestIdentity);
        }
        if request.grammar_constrained && request.chain_k != 1 {
            return Err(StreamingSupervisorError::GrammarChainKMustBeOne);
        }

        let key = request_key(&request.session_id, &request.req_id);
        if self.requests.contains_key(&key) {
            return Err(StreamingSupervisorError::DuplicateRequest);
        }
        self.requests.insert(key, RequestStream::new(request));
        Ok(())
    }

    /// Accept a worker frame into the authoritative stream history.
    ///
    /// Repeated sequence numbers are intentionally idempotent. A forward gap is
    /// not turned into guessed history: the frame is not committed, the returned
    /// disposition tells the consumer to query [`Self::session_status`], and a
    /// later worker death becomes a fail-closed failed session.
    pub fn observe_frame(
        &mut self,
        frame: &StreamFrameEnvelope,
    ) -> Result<StreamFrameDisposition, StreamingSupervisorError> {
        let state = self.request_mut(&frame.session_id, &frame.req_id)?;

        // The shared validator is cloned so a rejected identity cannot consume a
        // sequence number or accidentally create a terminal state.
        let mut candidate = state.validator.clone();
        let disposition = candidate.observe(frame)?;
        match disposition {
            StreamFrameDisposition::Duplicate => return Ok(StreamFrameDisposition::Duplicate),
            StreamFrameDisposition::Gap { expected, received } => {
                state.history_provable = false;
                return Ok(StreamFrameDisposition::Gap { expected, received });
            }
            StreamFrameDisposition::Accepted => {}
        }

        state.validate_expected_terminal(frame)?;
        state.validator = candidate;
        state.last_accepted_sequence = Some(frame.stream_seq);
        if let Some(terminal_state) = terminal_state(&frame.frame) {
            state.terminal_state = Some(terminal_state);
            state.cleanup_pending = true;
        }
        Ok(StreamFrameDisposition::Accepted)
    }

    /// Return the authoritative committed count and in-flight or terminal state.
    pub fn session_status(
        &self,
        session_id: &str,
        req_id: &str,
    ) -> Result<SessionStatus, StreamingSupervisorError> {
        Ok(self.request(session_id, req_id)?.status())
    }

    /// Abort at the last committed boundary and optionally retain that exact KV prefix.
    pub fn abort(
        &mut self,
        session_id: &str,
        req_id: &str,
        retention: RetentionPreflight,
        speculative_telemetry: Option<SpeculativeTelemetry>,
    ) -> Result<AbortOutcome, StreamingSupervisorError> {
        let (cancellation, terminal, retained_prefix) = {
            let state = self.request(session_id, req_id)?;
            if state.terminal_state.is_some() {
                return Err(StreamingSupervisorError::AlreadyTerminal);
            }
            let committed_token_count = state.validator.committed_token_count();
            let retained_prefix = preflight_retention(retention, committed_token_count)?;
            let terminal = terminal_frame(state, TerminalState::Aborted, speculative_telemetry)?;
            let cancellation = CancelledTransportResponse {
                req_id: state.request.req_id.clone(),
                generation_id: state.request.generation_id.clone(),
                committed_token_count,
            };
            (cancellation, terminal, retained_prefix)
        };

        let disposition = self.observe_frame(&terminal)?;
        debug_assert_eq!(disposition, StreamFrameDisposition::Accepted);
        let state = self.request_mut(session_id, req_id)?;
        state.retained_prefix = retained_prefix.clone();
        // A successful retention owns the resident prefix. Every other abort is
        // queued for the immediately following supervision-cycle cleanup.
        state.cleanup_pending = retained_prefix.is_none();

        Ok(AbortOutcome {
            cancellation,
            terminal,
            retained_prefix,
        })
    }

    /// Record a worker death without inventing unseen committed history.
    pub fn worker_died(
        &mut self,
        session_id: &str,
        req_id: &str,
        speculative_telemetry: Option<SpeculativeTelemetry>,
    ) -> Result<WorkerDeathOutcome, StreamingSupervisorError> {
        let terminal = {
            let state = self.request(session_id, req_id)?;
            if state.terminal_state.is_some() {
                return Err(StreamingSupervisorError::AlreadyTerminal);
            }
            if !state.history_provable {
                None
            } else {
                Some(terminal_frame(
                    state,
                    TerminalState::Failed,
                    speculative_telemetry,
                )?)
            }
        };

        if let Some(terminal) = terminal {
            let disposition = self.observe_frame(&terminal)?;
            debug_assert_eq!(disposition, StreamFrameDisposition::Accepted);
            return Ok(WorkerDeathOutcome::Terminal(Box::new(terminal)));
        }

        let state = self.request_mut(session_id, req_id)?;
        state.terminal_state = Some(TerminalState::Failed);
        state.cleanup_pending = true;
        Ok(WorkerDeathOutcome::FailedWithoutTerminal(state.status()))
    }

    /// Validate a continuation against the retained prefix, status, and artifact state.
    pub fn continuation_prefix(
        &self,
        session_id: &str,
        req_id: &str,
        retained_kv_session_id: &str,
        artifact_state: ArtifactServingState,
    ) -> Result<RetainedPrefix, StreamingSupervisorError> {
        let state = self.request(session_id, req_id)?;
        let status = state.status();
        validate_continuation(artifact_state, &status)
            .map_err(StreamingSupervisorError::ContinuationRefused)?;
        let prefix =
            state
                .retained_prefix
                .clone()
                .ok_or(StreamingSupervisorError::ContinuationRefused(
                    OwnedDecodeRefusal::RetainedKvUnavailable,
                ))?;
        if prefix.retained_kv_session_id != retained_kv_session_id {
            return Err(StreamingSupervisorError::RetainedSessionIdMismatch);
        }
        if prefix.retained_position != status.committed_token_count {
            return Err(StreamingSupervisorError::RetainedPrefixAccountingMismatch);
        }
        Ok(prefix)
    }

    /// Reclaim every non-retained terminal resource due by this supervision cycle.
    pub fn supervision_cycle(&mut self) -> SupervisionCycle {
        let mut cycle = SupervisionCycle::default();
        for state in self.requests.values_mut() {
            if state.cleanup_pending && state.retained_prefix.is_none() {
                state.cleanup_pending = false;
                cycle.reclaimed_requests += 1;
            }
        }
        cycle
    }

    /// Count non-retained terminal resources awaiting their required cleanup cycle.
    #[must_use]
    pub fn cleanup_pending_count(&self) -> usize {
        self.requests
            .values()
            .filter(|state| state.cleanup_pending && state.retained_prefix.is_none())
            .count()
    }

    fn request(
        &self,
        session_id: &str,
        req_id: &str,
    ) -> Result<&RequestStream, StreamingSupervisorError> {
        self.requests
            .get(&request_key(session_id, req_id))
            .ok_or(StreamingSupervisorError::UnknownRequest)
    }

    fn request_mut(
        &mut self,
        session_id: &str,
        req_id: &str,
    ) -> Result<&mut RequestStream, StreamingSupervisorError> {
        self.requests
            .get_mut(&request_key(session_id, req_id))
            .ok_or(StreamingSupervisorError::UnknownRequest)
    }
}

fn request_key(session_id: &str, req_id: &str) -> (String, String) {
    (session_id.to_string(), req_id.to_string())
}

fn terminal_state(frame: &StreamWorkerFrame) -> Option<TerminalState> {
    match frame {
        StreamWorkerFrame::Progress { .. } => None,
        StreamWorkerFrame::Final { terminal } | StreamWorkerFrame::Error { terminal } => {
            Some(terminal.terminal_state)
        }
    }
}

fn terminal_frame(
    state: &RequestStream,
    terminal_state: TerminalState,
    speculative_telemetry: Option<SpeculativeTelemetry>,
) -> Result<StreamFrameEnvelope, StreamingSupervisorError> {
    let committed_token_count = state.validator.committed_token_count();
    Ok(StreamFrameEnvelope::new(
        &state.request.req_id,
        &state.request.session_id,
        state.next_sequence()?,
        StreamWorkerFrame::Error {
            terminal: TerminalEnvelope {
                req_id: state.request.req_id.clone(),
                session_id: state.request.session_id.clone(),
                committed_token_count,
                tokens_emitted: committed_token_count,
                identity: state.request.identity.clone(),
                terminal_state,
                decode_mode: state.request.decode_mode,
                speculative_telemetry,
            },
        },
    ))
}

fn preflight_retention(
    retention: RetentionPreflight,
    committed_token_count: u32,
) -> Result<Option<RetainedPrefix>, StreamingSupervisorError> {
    match retention {
        RetentionPreflight::NotRequested | RetentionPreflight::Refused => Ok(None),
        RetentionPreflight::Ready {
            retained_kv_session_id,
            retained_position,
        } => {
            if retained_kv_session_id.is_empty() {
                return Err(StreamingSupervisorError::EmptyRetainedSessionId);
            }
            if retained_position != committed_token_count {
                return Err(StreamingSupervisorError::RetentionPositionMismatch {
                    expected: committed_token_count,
                    actual: retained_position,
                });
            }
            Ok(Some(RetainedPrefix {
                retained_kv_session_id,
                retained_position,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_core::{Fingerprint, ProgressFrame, OWNED_DECODE_ENVELOPE_V2_SCHEMA};

    fn identity() -> OneshotEnvelopeIdentity {
        OneshotEnvelopeIdentity {
            decode_fingerprint: Fingerprint("decode-fingerprint".to_string()),
            processing_fingerprint: Fingerprint("processing-fingerprint".to_string()),
            runtime_config_digest: "runtime-digest".to_string(),
            worker_generation: 7,
            derived_digest: Some("derived-digest".to_string()),
        }
    }

    fn request() -> StreamRequest {
        StreamRequest {
            req_id: "request-1".to_string(),
            session_id: "session-1".to_string(),
            generation_id: "generation-1".to_string(),
            identity: identity(),
            decode_mode: DecodeMode::Serial,
            grammar_constrained: false,
            chain_k: 1,
        }
    }

    fn progress(
        sequence: u64,
        token_ids: Vec<u32>,
        committed_token_count: u32,
    ) -> StreamFrameEnvelope {
        StreamFrameEnvelope::new(
            "request-1",
            "session-1",
            StreamSequence(sequence),
            StreamWorkerFrame::Progress {
                progress: ProgressFrame {
                    committed_token_ids: token_ids,
                    committed_token_count,
                    boundary: synapse_core::ProgressBoundary::Yield,
                },
            },
        )
    }

    fn completed(sequence: u64, committed_token_count: u32) -> StreamFrameEnvelope {
        StreamFrameEnvelope::new(
            "request-1",
            "session-1",
            StreamSequence(sequence),
            StreamWorkerFrame::Final {
                terminal: TerminalEnvelope {
                    req_id: "request-1".to_string(),
                    session_id: "session-1".to_string(),
                    committed_token_count,
                    tokens_emitted: committed_token_count,
                    identity: identity(),
                    terminal_state: TerminalState::Completed,
                    decode_mode: DecodeMode::Serial,
                    speculative_telemetry: None,
                },
            },
        )
    }

    #[test]
    fn progress_is_ordered_idempotent_and_has_one_terminal() {
        let mut supervisor = StreamingSupervisor::default();
        supervisor.begin(request()).unwrap();

        let first = progress(1, vec![10, 11], 2);
        assert_eq!(
            supervisor.observe_frame(&first),
            Ok(StreamFrameDisposition::Accepted)
        );
        assert_eq!(
            supervisor.observe_frame(&first),
            Ok(StreamFrameDisposition::Duplicate)
        );
        assert_eq!(
            supervisor.observe_frame(&progress(2, vec![12], 3)),
            Ok(StreamFrameDisposition::Accepted)
        );
        assert_eq!(
            supervisor.observe_frame(&completed(3, 3)),
            Ok(StreamFrameDisposition::Accepted)
        );

        let status = supervisor.session_status("session-1", "request-1").unwrap();
        assert_eq!(status.committed_token_count, 3);
        assert_eq!(
            status.state,
            SessionStatusState::Terminal(TerminalState::Completed)
        );
        assert_eq!(
            supervisor.observe_frame(&completed(4, 3)),
            Err(StreamingSupervisorError::Envelope(
                EnvelopeValidationError::FrameAfterTerminal
            ))
        );
    }

    #[test]
    fn terminal_status_recovers_lost_delivery_with_its_committed_count() {
        let mut supervisor = StreamingSupervisor::default();
        supervisor.begin(request()).unwrap();
        supervisor.observe_frame(&progress(1, vec![10], 1)).unwrap();
        // A client can lose this terminal; it is already committed into the
        // supervisor's authority before stream delivery is attempted.
        supervisor.observe_frame(&completed(2, 1)).unwrap();

        assert_eq!(
            supervisor.session_status("session-1", "request-1").unwrap(),
            SessionStatus {
                session_id: "session-1".to_string(),
                req_id: "request-1".to_string(),
                committed_token_count: 1,
                state: SessionStatusState::Terminal(TerminalState::Completed),
                retained_kv_session_id: None,
            }
        );
    }

    #[test]
    fn gap_fails_closed_after_worker_death_and_refuses_continuation() {
        let mut supervisor = StreamingSupervisor::default();
        supervisor.begin(request()).unwrap();

        assert_eq!(
            supervisor.observe_frame(&progress(2, vec![10], 1)),
            Ok(StreamFrameDisposition::Gap {
                expected: StreamSequence::FIRST,
                received: StreamSequence(2),
            })
        );
        assert_eq!(
            supervisor.worker_died("session-1", "request-1", None),
            Ok(WorkerDeathOutcome::FailedWithoutTerminal(SessionStatus {
                session_id: "session-1".to_string(),
                req_id: "request-1".to_string(),
                committed_token_count: 0,
                state: SessionStatusState::Terminal(TerminalState::Failed),
                retained_kv_session_id: None,
            }))
        );
        assert_eq!(
            supervisor.continuation_prefix(
                "session-1",
                "request-1",
                "retained-session",
                ArtifactServingState::Approved,
            ),
            Err(StreamingSupervisorError::ContinuationRefused(
                OwnedDecodeRefusal::FailedSessionContinuation
            ))
        );
    }

    #[test]
    fn worker_death_emits_failed_terminal_when_ordered_history_is_provable() {
        let mut supervisor = StreamingSupervisor::default();
        supervisor.begin(request()).unwrap();
        supervisor
            .observe_frame(&progress(1, vec![10, 11], 2))
            .unwrap();

        let outcome = supervisor
            .worker_died("session-1", "request-1", None)
            .unwrap();
        let WorkerDeathOutcome::Terminal(terminal) = outcome else {
            panic!("ordered history must produce a terminal accounting frame");
        };
        assert_eq!(terminal.protocol, OWNED_DECODE_ENVELOPE_V2_SCHEMA);
        assert_eq!(terminal.stream_seq, StreamSequence(2));
        let StreamWorkerFrame::Error { terminal } = terminal.frame else {
            panic!("worker death must use an error terminal");
        };
        assert_eq!(terminal.terminal_state, TerminalState::Failed);
        assert_eq!(terminal.committed_token_count, 2);
        assert_eq!(terminal.tokens_emitted, 2);
        assert_eq!(
            supervisor
                .session_status("session-1", "request-1")
                .unwrap()
                .state,
            SessionStatusState::Terminal(TerminalState::Failed)
        );
    }

    #[test]
    fn abort_preflights_retention_at_the_committed_prefix() {
        let mut supervisor = StreamingSupervisor::default();
        supervisor.begin(request()).unwrap();
        supervisor
            .observe_frame(&progress(1, vec![10, 11], 2))
            .unwrap();

        let outcome = supervisor
            .abort(
                "session-1",
                "request-1",
                RetentionPreflight::Ready {
                    retained_kv_session_id: "retained-session-1".to_string(),
                    retained_position: 2,
                },
                None,
            )
            .unwrap();
        assert_eq!(outcome.cancellation.generation_id, "generation-1");
        assert_eq!(outcome.cancellation.committed_token_count, 2);
        assert_eq!(
            outcome.retained_prefix,
            Some(RetainedPrefix {
                retained_kv_session_id: "retained-session-1".to_string(),
                retained_position: 2,
            })
        );
        let status = supervisor.session_status("session-1", "request-1").unwrap();
        outcome
            .cancellation
            .validate_against_status(&status)
            .unwrap();
        assert_eq!(
            status.retained_kv_session_id.as_deref(),
            Some("retained-session-1")
        );
        assert_eq!(
            supervisor.continuation_prefix(
                "session-1",
                "request-1",
                "retained-session-1",
                ArtifactServingState::Approved,
            ),
            Ok(RetainedPrefix {
                retained_kv_session_id: "retained-session-1".to_string(),
                retained_position: 2,
            })
        );
        assert_eq!(supervisor.cleanup_pending_count(), 0);
    }

    #[test]
    fn retention_refusal_reclaims_resources_in_one_supervision_cycle() {
        let mut supervisor = StreamingSupervisor::default();
        supervisor.begin(request()).unwrap();
        supervisor.observe_frame(&progress(1, vec![10], 1)).unwrap();

        supervisor
            .abort("session-1", "request-1", RetentionPreflight::Refused, None)
            .unwrap();
        assert_eq!(supervisor.cleanup_pending_count(), 1);
        assert_eq!(
            supervisor.supervision_cycle(),
            SupervisionCycle {
                reclaimed_requests: 1
            }
        );
        assert_eq!(supervisor.cleanup_pending_count(), 0);
        // Cleanup removes resources, not recovery authority.
        assert_eq!(
            supervisor
                .session_status("session-1", "request-1")
                .unwrap()
                .committed_token_count,
            1
        );
    }

    #[test]
    fn terminal_identity_and_grammar_chain_are_not_relaxed() {
        let mut supervisor = StreamingSupervisor::default();
        let mut constrained = request();
        constrained.grammar_constrained = true;
        constrained.chain_k = 2;
        assert_eq!(
            supervisor.begin(constrained),
            Err(StreamingSupervisorError::GrammarChainKMustBeOne)
        );

        supervisor.begin(request()).unwrap();
        let mut wrong_identity = completed(1, 0);
        let StreamWorkerFrame::Final { terminal } = &mut wrong_identity.frame else {
            unreachable!();
        };
        terminal.identity.worker_generation = 8;
        assert_eq!(
            supervisor.observe_frame(&wrong_identity),
            Err(StreamingSupervisorError::TerminalIdentityMismatch)
        );
        assert_eq!(
            supervisor
                .session_status("session-1", "request-1")
                .unwrap()
                .state,
            SessionStatusState::InFlight
        );
    }
}
