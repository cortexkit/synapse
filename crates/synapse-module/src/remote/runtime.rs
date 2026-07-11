use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, RwLock},
};

use serde::{Deserialize, Serialize};
use subc_client_rs::async_trait;
use synapse_core::{Fingerprint, NumericProfileId, StableError};
use tokio::sync::Notify;

use super::{classify::FailureClass, ContinuityCheck, ContinuityError};
use crate::store::{
    AssuranceClass, CertificationKey, CertificationRow, SynapseStore, SynapseStoreError,
};

const ESTIMATOR_WINDOW_MS: u64 = 30 * 60 * 1_000;
const ESTIMATOR_CAPACITY: usize = 256;
const ESTIMATOR_MIN_SAMPLES: usize = 8;
const CREDENTIAL_EXPIRY_MARGIN_MS: u64 = 60_000;
const CALIBRATION_RUNS: usize = 5;
const PERFECT_DRIFT_GATE: f64 = 0.9999;

/// Admission class for remote I/O. These permits are independent from local GPU lane quanta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RemoteClass {
    Interactive,
    Bulk,
}

#[derive(Debug)]
struct PoolState {
    active: usize,
    active_bulk: usize,
    waiting_interactive: usize,
}

#[derive(Debug)]
struct PoolInner {
    max_concurrency: usize,
    state: Mutex<PoolState>,
    turnover: Notify,
}

#[derive(Clone, Debug)]
pub(super) struct ProviderPool {
    inner: Arc<PoolInner>,
}

impl ProviderPool {
    pub(super) fn new(max_concurrency: usize) -> Result<Self, RuntimeError> {
        if max_concurrency == 0 {
            return Err(RuntimeError::invalid_config(
                "provider max_concurrency must be at least one",
            ));
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                max_concurrency,
                state: Mutex::new(PoolState {
                    active: 0,
                    active_bulk: 0,
                    waiting_interactive: 0,
                }),
                turnover: Notify::new(),
            }),
        })
    }

    pub(super) async fn acquire(&self, class: RemoteClass) -> ProviderPermit {
        let mut waiter = InteractiveWaiter::new(self, class);
        loop {
            let notified = self.inner.turnover.notified();
            {
                let mut state = self
                    .inner
                    .state
                    .lock()
                    .expect("provider pool lock poisoned");
                let bulk_capacity = if self.inner.max_concurrency >= 2 {
                    self.inner.max_concurrency - 1
                } else {
                    1
                };
                let may_acquire = match class {
                    RemoteClass::Interactive => state.active < self.inner.max_concurrency,
                    RemoteClass::Bulk => {
                        state.active_bulk < bulk_capacity && state.waiting_interactive == 0
                    }
                };
                if may_acquire {
                    state.active += 1;
                    if class == RemoteClass::Bulk {
                        state.active_bulk += 1;
                    }
                    waiter.finish(&mut state);
                    return ProviderPermit {
                        pool: Arc::clone(&self.inner),
                        class,
                    };
                }
            }
            notified.await;
        }
    }

    /// Sizes a bulk sub-batch so its estimated latency does not exceed the turnover target.
    pub(super) fn subbatch_tokens(
        &self,
        total_tokens: u64,
        estimated_ms: u64,
        target_subbatch_ms: u64,
    ) -> u64 {
        if total_tokens == 0 {
            return 0;
        }
        if estimated_ms == 0 || estimated_ms <= target_subbatch_ms {
            return total_tokens;
        }
        let numerator = u128::from(total_tokens) * u128::from(target_subbatch_ms.max(1));
        let sized = numerator / u128::from(estimated_ms);
        u64::try_from(sized)
            .unwrap_or(u64::MAX)
            .clamp(1, total_tokens)
    }
}

struct InteractiveWaiter<'a> {
    pool: &'a ProviderPool,
    registered: bool,
}

impl<'a> InteractiveWaiter<'a> {
    fn new(pool: &'a ProviderPool, class: RemoteClass) -> Self {
        let registered = class == RemoteClass::Interactive;
        if registered {
            let mut state = pool
                .inner
                .state
                .lock()
                .expect("provider pool lock poisoned");
            state.waiting_interactive += 1;
        }
        Self { pool, registered }
    }

    fn finish(&mut self, state: &mut PoolState) {
        if self.registered {
            state.waiting_interactive = state.waiting_interactive.saturating_sub(1);
            self.registered = false;
        }
    }
}

impl Drop for InteractiveWaiter<'_> {
    fn drop(&mut self) {
        if self.registered {
            if let Ok(mut state) = self.pool.inner.state.lock() {
                state.waiting_interactive = state.waiting_interactive.saturating_sub(1);
            }
            self.pool.inner.turnover.notify_waiters();
        }
    }
}

#[derive(Debug)]
pub(super) struct ProviderPermit {
    pool: Arc<PoolInner>,
    class: RemoteClass,
}

impl Drop for ProviderPermit {
    fn drop(&mut self) {
        if let Ok(mut state) = self.pool.state.lock() {
            state.active = state.active.saturating_sub(1);
            if self.class == RemoteClass::Bulk {
                state.active_bulk = state.active_bulk.saturating_sub(1);
            }
        }
        self.pool.turnover.notify_waiters();
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct BreakerKey {
    pub provider: String,
    pub deployment: String,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BreakerConfig {
    pub failure_threshold: u32,
    pub cooldown_ms: u64,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            cooldown_ms: 30_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BreakerStateSnapshot {
    Closed,
    Open { retry_after_ms: u64 },
    HalfOpen,
}

#[derive(Clone, Copy, Debug)]
enum BreakerState {
    Closed { consecutive_failures: u32 },
    Open { opened_at_ms: u64 },
    HalfOpen { probe_in_flight: bool },
}

#[derive(Clone, Copy, Debug)]
struct BreakerEntry {
    epoch: u64,
    state: BreakerState,
}

#[derive(Debug)]
pub(super) struct BreakerLease {
    key: BreakerKey,
    epoch: u64,
    half_open_probe: bool,
}

impl BreakerLease {
    pub(super) fn is_half_open_probe(&self) -> bool {
        self.half_open_probe
    }
}

#[derive(Debug)]
pub(super) struct CircuitBreaker {
    config: BreakerConfig,
    entries: Mutex<HashMap<BreakerKey, BreakerEntry>>,
}

impl CircuitBreaker {
    pub(super) fn new(config: BreakerConfig) -> Result<Self, RuntimeError> {
        if config.failure_threshold == 0 || config.cooldown_ms == 0 {
            return Err(RuntimeError::invalid_config(
                "breaker failure_threshold and cooldown_ms must be non-zero",
            ));
        }
        Ok(Self {
            config,
            entries: Mutex::new(HashMap::new()),
        })
    }

    pub(super) fn admit(
        &self,
        key: &BreakerKey,
        now_ms: u64,
    ) -> Result<BreakerLease, RuntimeError> {
        let mut entries = self.entries.lock().expect("breaker lock poisoned");
        let entry = entries.entry(key.clone()).or_insert(BreakerEntry {
            epoch: 0,
            state: BreakerState::Closed {
                consecutive_failures: 0,
            },
        });
        let half_open_probe = match entry.state {
            BreakerState::Closed { .. } => false,
            BreakerState::Open { opened_at_ms } => {
                let elapsed = now_ms.saturating_sub(opened_at_ms);
                if elapsed < self.config.cooldown_ms {
                    return Err(RuntimeError::provider_unavailable(
                        self.config.cooldown_ms - elapsed,
                        &key.provider,
                    ));
                }
                entry.epoch = entry.epoch.wrapping_add(1);
                entry.state = BreakerState::HalfOpen {
                    probe_in_flight: true,
                };
                true
            }
            BreakerState::HalfOpen {
                probe_in_flight: true,
            } => {
                return Err(RuntimeError::provider_unavailable(
                    self.config.cooldown_ms,
                    &key.provider,
                ))
            }
            BreakerState::HalfOpen {
                probe_in_flight: false,
            } => {
                entry.state = BreakerState::HalfOpen {
                    probe_in_flight: true,
                };
                true
            }
        };
        Ok(BreakerLease {
            key: key.clone(),
            epoch: entry.epoch,
            half_open_probe,
        })
    }

    pub(super) fn record_success(&self, lease: &BreakerLease) {
        let mut entries = self.entries.lock().expect("breaker lock poisoned");
        let Some(entry) = entries.get_mut(&lease.key) else {
            return;
        };
        if entry.epoch != lease.epoch {
            return;
        }
        entry.epoch = entry.epoch.wrapping_add(1);
        entry.state = BreakerState::Closed {
            consecutive_failures: 0,
        };
    }

    /// Only classifier-transient failures feed the transport breaker. Pacing and permanent
    /// responses are deliberately excluded, including every HTTP 429.
    pub(super) fn record_failure(
        &self,
        lease: &BreakerLease,
        classification: FailureClass,
        now_ms: u64,
    ) {
        if classification != FailureClass::Transient {
            return;
        }
        let mut entries = self.entries.lock().expect("breaker lock poisoned");
        let Some(entry) = entries.get_mut(&lease.key) else {
            return;
        };
        if entry.epoch != lease.epoch {
            return;
        }
        let failures = match entry.state {
            BreakerState::Closed {
                consecutive_failures,
            } => consecutive_failures.saturating_add(1),
            BreakerState::HalfOpen { .. } => self.config.failure_threshold,
            BreakerState::Open { .. } => return,
        };
        entry.epoch = entry.epoch.wrapping_add(1);
        entry.state = if failures >= self.config.failure_threshold {
            BreakerState::Open {
                opened_at_ms: now_ms,
            }
        } else {
            BreakerState::Closed {
                consecutive_failures: failures,
            }
        };
    }

    pub(super) fn state(&self, key: &BreakerKey, now_ms: u64) -> BreakerStateSnapshot {
        let entries = self.entries.lock().expect("breaker lock poisoned");
        match entries.get(key).map(|entry| entry.state) {
            None | Some(BreakerState::Closed { .. }) => BreakerStateSnapshot::Closed,
            Some(BreakerState::Open { opened_at_ms }) => BreakerStateSnapshot::Open {
                retry_after_ms: self
                    .config
                    .cooldown_ms
                    .saturating_sub(now_ms.saturating_sub(opened_at_ms)),
            },
            Some(BreakerState::HalfOpen { .. }) => BreakerStateSnapshot::HalfOpen,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum TokenBucket {
    UpTo1k,
    UpTo8k,
    UpTo64k,
    Above64k,
}

impl TokenBucket {
    pub(super) fn for_tokens(tokens: u64) -> Self {
        match tokens {
            0..=1_000 => Self::UpTo1k,
            1_001..=8_000 => Self::UpTo8k,
            8_001..=64_000 => Self::UpTo64k,
            _ => Self::Above64k,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EstimateKey {
    provider: String,
    operation: String,
    bucket: TokenBucket,
}

#[derive(Clone, Copy, Debug)]
struct TimedSample {
    observed_at_ms: u64,
    duration_ms: u64,
}

#[derive(Default, Debug)]
struct EstimateWindow {
    samples: VecDeque<TimedSample>,
    censors: VecDeque<TimedSample>,
}

#[derive(Debug, Default)]
pub(super) struct LatencyEstimator {
    windows: Mutex<HashMap<EstimateKey, EstimateWindow>>,
}

impl LatencyEstimator {
    pub(super) fn observe_latency(
        &self,
        provider: &str,
        operation: &str,
        tokens: u64,
        duration_ms: u64,
        now_ms: u64,
    ) {
        self.observe(provider, operation, tokens, duration_ms, now_ms, false);
    }

    pub(super) fn observe_censored_timeout(
        &self,
        provider: &str,
        operation: &str,
        tokens: u64,
        duration_ms: u64,
        now_ms: u64,
    ) {
        self.observe(provider, operation, tokens, duration_ms, now_ms, true);
    }

    fn observe(
        &self,
        provider: &str,
        operation: &str,
        tokens: u64,
        duration_ms: u64,
        now_ms: u64,
        censored: bool,
    ) {
        let key = EstimateKey {
            provider: provider.to_string(),
            operation: operation.to_string(),
            bucket: TokenBucket::for_tokens(tokens),
        };
        let mut windows = self.windows.lock().expect("estimator lock poisoned");
        let window = windows.entry(key).or_default();
        expire_window(window, now_ms);
        let samples = if censored {
            &mut window.censors
        } else {
            &mut window.samples
        };
        samples.push_back(TimedSample {
            observed_at_ms: now_ms,
            duration_ms,
        });
        while samples.len() > ESTIMATOR_CAPACITY {
            samples.pop_front();
        }
    }

    pub(super) fn estimate_ms(
        &self,
        provider: &str,
        operation: &str,
        tokens: u64,
        cold_estimate_ms: u64,
        now_ms: u64,
    ) -> u64 {
        let bucket = TokenBucket::for_tokens(tokens);
        let mut windows = self.windows.lock().expect("estimator lock poisoned");
        for (key, window) in windows.iter_mut() {
            if key.provider == provider && key.operation == operation {
                expire_window(window, now_ms);
            }
        }

        let bucket_samples = windows
            .get(&EstimateKey {
                provider: provider.to_string(),
                operation: operation.to_string(),
                bucket,
            })
            .map(|window| sample_durations(&window.samples))
            .unwrap_or_default();
        let chosen = if bucket_samples.len() >= ESTIMATOR_MIN_SAMPLES {
            nearest_rank_p90(bucket_samples)
        } else {
            let merged = windows
                .iter()
                .filter(|(key, _)| key.provider == provider && key.operation == operation)
                .flat_map(|(_, window)| window.samples.iter().map(|sample| sample.duration_ms))
                .collect::<Vec<_>>();
            if merged.len() >= ESTIMATOR_MIN_SAMPLES {
                nearest_rank_p90(merged)
            } else {
                cold_estimate_ms
            }
        };
        let censor_floor = windows
            .get(&EstimateKey {
                provider: provider.to_string(),
                operation: operation.to_string(),
                bucket,
            })
            .and_then(|window| window.censors.iter().map(|sample| sample.duration_ms).max())
            .unwrap_or(0);
        chosen.max(censor_floor)
    }
}

fn expire_window(window: &mut EstimateWindow, now_ms: u64) {
    let cutoff = now_ms.saturating_sub(ESTIMATOR_WINDOW_MS);
    window
        .samples
        .retain(|sample| sample.observed_at_ms >= cutoff && sample.observed_at_ms <= now_ms);
    window
        .censors
        .retain(|sample| sample.observed_at_ms >= cutoff && sample.observed_at_ms <= now_ms);
}

fn sample_durations(samples: &VecDeque<TimedSample>) -> Vec<u64> {
    samples.iter().map(|sample| sample.duration_ms).collect()
}

fn nearest_rank_p90(mut values: Vec<u64>) -> u64 {
    debug_assert!(!values.is_empty());
    values.sort_unstable();
    let rank = values.len().saturating_mul(9).div_ceil(10).max(1);
    values[rank - 1]
}

pub(crate) struct CredentialToken {
    secret: String,
    pub expires_at_ms: u64,
    pub record_version: u64,
}

impl CredentialToken {
    pub(super) fn new(secret: String, expires_at_ms: u64, record_version: u64) -> Self {
        Self {
            secret,
            expires_at_ms,
            record_version,
        }
    }

    pub(super) fn expose_secret(&self) -> &str {
        &self.secret
    }
}

impl std::fmt::Debug for CredentialToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CredentialToken")
            .field("secret", &"[REDACTED]")
            .field("expires_at_ms", &self.expires_at_ms)
            .field("record_version", &self.record_version)
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum VaultError {
    NotFound,
    MalformedHandle,
    MalformedHandlesFile,
    Unreachable,
    VaultLocked,
    NeedsReauth,
}

#[async_trait]
pub(crate) trait VaultCredentialClient: Send + Sync {
    async fn fetch(
        &self,
        logical_handle: &str,
        min_ttl_ms: u64,
    ) -> Result<CredentialToken, VaultError>;

    async fn report_auth_failure(
        &self,
        logical_handle: &str,
        provider_status: u16,
        record_version: u64,
    );
}

#[derive(Debug)]
pub(super) enum CredentialDisposition {
    Reject(RuntimeError),
    PauseJob,
}

pub(super) struct CredentialManager {
    client: Arc<dyn VaultCredentialClient>,
}

pub(super) struct JobCredentialRequest<'a> {
    pub job_id: &'a str,
    pub logical_handle: &'a str,
    pub configured_attempt_timeout_ms: u64,
    pub remaining_deadline_ms: u64,
    pub now_ms: u64,
    pub resume_window_ms: u64,
}

impl CredentialManager {
    pub(super) fn new(client: Arc<dyn VaultCredentialClient>) -> Self {
        Self { client }
    }

    pub(super) async fn acquire(
        &self,
        logical_handle: &str,
        configured_attempt_timeout_ms: u64,
        remaining_deadline_ms: u64,
    ) -> Result<CredentialToken, CredentialDisposition> {
        let effective_attempt_timeout_ms = configured_attempt_timeout_ms.min(remaining_deadline_ms);
        let min_ttl_ms = effective_attempt_timeout_ms.saturating_add(CREDENTIAL_EXPIRY_MARGIN_MS);
        self.client
            .fetch(logical_handle, min_ttl_ms)
            .await
            .map_err(|error| match error {
                VaultError::VaultLocked | VaultError::NeedsReauth => {
                    CredentialDisposition::PauseJob
                }
                VaultError::MalformedHandlesFile
                | VaultError::MalformedHandle
                | VaultError::NotFound => {
                    CredentialDisposition::Reject(RuntimeError::credential_config_invalid(
                        "credential handle configuration is invalid",
                    ))
                }
                VaultError::Unreachable => CredentialDisposition::Reject(
                    RuntimeError::provider_unavailable(1_000, "credential vault"),
                ),
            })
    }

    pub(super) async fn acquire_for_job(
        &self,
        store: &SynapseStore,
        request: JobCredentialRequest<'_>,
    ) -> Result<Option<CredentialToken>, RuntimeError> {
        match self
            .acquire(
                request.logical_handle,
                request.configured_attempt_timeout_ms,
                request.remaining_deadline_ms,
            )
            .await
        {
            Ok(token) => Ok(Some(token)),
            Err(CredentialDisposition::PauseJob) => {
                store
                    .pause_job_needs_reauth(
                        request.job_id,
                        request.logical_handle,
                        request.now_ms,
                        request.resume_window_ms,
                    )
                    .map_err(RuntimeError::store)?;
                Ok(None)
            }
            Err(CredentialDisposition::Reject(error)) => Err(error),
        }
    }

    /// This is called only after the provider retry loop has terminated.
    pub(super) async fn report_terminal_auth_failure(
        &self,
        logical_handle: &str,
        provider_status: u16,
        record_version: u64,
    ) {
        if matches!(provider_status, 401 | 403) {
            self.client
                .report_auth_failure(logical_handle, provider_status, record_version)
                .await;
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct SentinelProfile {
    pub synapse_model_id: String,
    pub machine_profile_hash: String,
    pub remote_profile_hash: String,
    pub identity_revision: String,
    pub fingerprint: Fingerprint,
    pub numeric_profile_id: NumericProfileId,
    pub sentinel_texts: Vec<String>,
    pub drift_gate_min: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct SentinelCalibration {
    pub sentinel_texts: Vec<String>,
    pub baseline_vectors: Vec<Vec<f32>>,
    pub calibration_floor: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DriftState {
    Healthy,
    Suspect,
    Quarantined,
}

#[async_trait]
pub(super) trait SentinelEmbedder: Send + Sync {
    async fn embed_sentinels(
        &self,
        profile: &SentinelProfile,
        logical_handle: Option<&str>,
    ) -> Result<Vec<Vec<f32>>, RuntimeError>;
}

struct UnconfiguredSentinelEmbedder;

#[async_trait]
impl SentinelEmbedder for UnconfiguredSentinelEmbedder {
    async fn embed_sentinels(
        &self,
        profile: &SentinelProfile,
        _logical_handle: Option<&str>,
    ) -> Result<Vec<Vec<f32>>, RuntimeError> {
        Err(RuntimeError::provider_unavailable(
            1_000,
            &profile.synapse_model_id,
        ))
    }
}

pub(crate) struct SentinelContinuityCheck {
    store: Arc<SynapseStore>,
    embedder: Arc<dyn SentinelEmbedder>,
    profiles: RwLock<HashMap<String, SentinelProfile>>,
    states: Mutex<HashMap<String, DriftState>>,
}

impl SentinelContinuityCheck {
    pub(crate) fn empty(store: Arc<SynapseStore>) -> Self {
        Self::new(
            store,
            Arc::new(UnconfiguredSentinelEmbedder) as Arc<dyn SentinelEmbedder>,
        )
    }

    pub(super) fn new(store: Arc<SynapseStore>, embedder: Arc<dyn SentinelEmbedder>) -> Self {
        Self {
            store,
            embedder,
            profiles: RwLock::new(HashMap::new()),
            states: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn register_profile(&self, profile: SentinelProfile) {
        self.profiles
            .write()
            .expect("sentinel profiles lock poisoned")
            .insert(profile.synapse_model_id.clone(), profile);
    }

    pub(super) fn state(&self, synapse_model_id: &str) -> DriftState {
        self.states
            .lock()
            .expect("sentinel states lock poisoned")
            .get(synapse_model_id)
            .copied()
            .unwrap_or(DriftState::Healthy)
    }

    pub(super) async fn calibrate(
        &self,
        profile: &SentinelProfile,
        logical_handle: Option<&str>,
    ) -> Result<SentinelCalibration, RuntimeError> {
        validate_drift_config(profile)?;
        if profile.sentinel_texts.is_empty() {
            return Err(RuntimeError::protocol(
                "sentinel calibration requires at least one sentinel",
            ));
        }
        let mut runs = Vec::with_capacity(CALIBRATION_RUNS);
        for _ in 0..CALIBRATION_RUNS {
            let vectors = self
                .embedder
                .embed_sentinels(profile, logical_handle)
                .await?;
            validate_sentinel_vectors(profile.sentinel_texts.len(), &vectors)?;
            runs.push(vectors);
        }
        let calibration_floor = calibration_floor(&runs)?;
        if calibration_floor < (1.0 + profile.drift_gate_min) / 2.0 {
            return Err(RuntimeError::protocol(
                "provider self-noise exceeds half the configured drift budget",
            ));
        }
        let baseline_vectors = mean_vectors(&runs)?;
        Ok(SentinelCalibration {
            sentinel_texts: profile.sentinel_texts.clone(),
            baseline_vectors,
            calibration_floor,
        })
    }

    pub(super) async fn calibrate_and_store(
        &self,
        profile: &SentinelProfile,
        logical_handle: Option<&str>,
        certified_at_ms: u64,
        module_generation: u64,
    ) -> Result<CertificationRow, RuntimeError> {
        let calibration = self.calibrate(profile, logical_handle).await?;
        let row = CertificationRow {
            assurance_class: AssuranceClass::Declared,
            key: CertificationKey::Declared {
                machine_profile_hash: profile.machine_profile_hash.clone(),
                remote_profile_hash: profile.remote_profile_hash.clone(),
                identity_revision: profile.identity_revision.clone(),
            },
            numeric_profile_id: profile.numeric_profile_id.clone(),
            fingerprint: profile.fingerprint.clone(),
            certified_at_ms,
            os_build: String::new(),
            module_generation,
            evidence: serde_json::json!({"remote_sentinel": calibration}),
        };
        self.store
            .store_cert_row(&row)
            .map_err(RuntimeError::store)?;
        self.register_profile(profile.clone());
        Ok(row)
    }

    pub(super) async fn check_profile(
        &self,
        profile: &SentinelProfile,
        logical_handle: Option<&str>,
    ) -> Result<(), RuntimeError> {
        if self.state(&profile.synapse_model_id) == DriftState::Quarantined {
            return Err(RuntimeError::identity_drift(&profile.synapse_model_id));
        }
        validate_drift_config(profile)?;
        let mut row = self
            .store
            .get_declared_cert_row(
                &profile.machine_profile_hash,
                &profile.remote_profile_hash,
                &profile.identity_revision,
                &profile.fingerprint,
            )
            .map_err(RuntimeError::store)?
            .ok_or_else(|| RuntimeError {
                stable: StableError::not_certified(),
                message: format!(
                    "remote profile '{}' has not been calibrated on this machine",
                    profile.synapse_model_id
                ),
            })?;
        if row
            .evidence
            .get("remote_sentinel_quarantined")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            self.set_state(&profile.synapse_model_id, DriftState::Quarantined);
            return Err(RuntimeError::identity_drift(&profile.synapse_model_id));
        }
        let calibration_value = row
            .evidence
            .get("remote_sentinel")
            .cloned()
            .ok_or_else(|| {
                RuntimeError::protocol("declared certification has no sentinel baseline")
            })?;
        let calibration: SentinelCalibration =
            serde_json::from_value(calibration_value).map_err(|error| {
                RuntimeError::protocol(format!("invalid sentinel baseline: {error}"))
            })?;
        validate_sentinel_vectors(profile.sentinel_texts.len(), &calibration.baseline_vectors)?;
        let derived_gate = (2.0 * calibration.calibration_floor - 1.0).min(PERFECT_DRIFT_GATE);
        let gate = derived_gate.max(profile.drift_gate_min);

        let first = self
            .embedder
            .embed_sentinels(profile, logical_handle)
            .await?;
        if sentinel_run_passes(&calibration.baseline_vectors, &first, gate)? {
            self.set_state(&profile.synapse_model_id, DriftState::Healthy);
            return Ok(());
        }

        self.set_state(&profile.synapse_model_id, DriftState::Suspect);
        let confirmation = self
            .embedder
            .embed_sentinels(profile, logical_handle)
            .await?;
        if sentinel_run_passes(&calibration.baseline_vectors, &confirmation, gate)? {
            self.set_state(&profile.synapse_model_id, DriftState::Healthy);
            return Ok(());
        }
        self.set_state(&profile.synapse_model_id, DriftState::Quarantined);
        let evidence = row.evidence.as_object_mut().ok_or_else(|| {
            RuntimeError::protocol("declared certification evidence must be an object")
        })?;
        evidence.insert(
            "remote_sentinel_quarantined".to_string(),
            serde_json::Value::Bool(true),
        );
        self.store
            .store_cert_row(&row)
            .map_err(RuntimeError::store)?;
        Err(RuntimeError::identity_drift(&profile.synapse_model_id))
    }

    fn set_state(&self, synapse_model_id: &str, state: DriftState) {
        self.states
            .lock()
            .expect("sentinel states lock poisoned")
            .insert(synapse_model_id.to_string(), state);
    }
}

#[async_trait]
impl ContinuityCheck for SentinelContinuityCheck {
    async fn check(
        &self,
        _request_digest: &str,
        synapse_model_id: &str,
        logical_handle: Option<&str>,
    ) -> Result<(), ContinuityError> {
        let profile = self
            .profiles
            .read()
            .expect("sentinel profiles lock poisoned")
            .get(synapse_model_id)
            .cloned();
        let Some(profile) = profile else {
            // Checkpoint continuity also runs for local durable jobs; only registered remote
            // profiles have provider identity that requires a sentinel call.
            return Ok(());
        };
        self.check_profile(&profile, logical_handle)
            .await
            .map_err(|error| ContinuityError::new(error.message))
    }
}

fn validate_drift_config(profile: &SentinelProfile) -> Result<(), RuntimeError> {
    if !profile.drift_gate_min.is_finite()
        || profile.drift_gate_min < -1.0
        || profile.drift_gate_min > PERFECT_DRIFT_GATE
    {
        return Err(RuntimeError::invalid_config(
            "drift_gate_min must be finite and within [-1, 0.9999]",
        ));
    }
    Ok(())
}

fn validate_sentinel_vectors(expected: usize, vectors: &[Vec<f32>]) -> Result<(), RuntimeError> {
    if vectors.len() != expected || vectors.is_empty() {
        return Err(RuntimeError::protocol(format!(
            "provider returned {} sentinel vectors for {expected} inputs",
            vectors.len()
        )));
    }
    let dimensions = vectors[0].len();
    if dimensions == 0 || vectors.iter().any(|vector| vector.len() != dimensions) {
        return Err(RuntimeError::protocol(
            "provider returned empty or ragged sentinel vectors",
        ));
    }
    for vector in vectors {
        if vector.iter().any(|value| !value.is_finite())
            || vector
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                == 0.0
        {
            return Err(RuntimeError::protocol(
                "provider returned a non-finite or zero-norm sentinel vector",
            ));
        }
    }
    Ok(())
}

fn calibration_floor(runs: &[Vec<Vec<f32>>]) -> Result<f64, RuntimeError> {
    let mut floor = 1.0_f64;
    for left in 0..runs.len() {
        for right in (left + 1)..runs.len() {
            for (left_vector, right_vector) in runs[left].iter().zip(&runs[right]) {
                floor = floor.min(cosine(left_vector, right_vector)?);
            }
        }
    }
    Ok(floor)
}

fn mean_vectors(runs: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<f32>>, RuntimeError> {
    let sentinel_count = runs.first().map_or(0, Vec::len);
    let dimensions = runs.first().and_then(|run| run.first()).map_or(0, Vec::len);
    let mut means = vec![vec![0.0_f32; dimensions]; sentinel_count];
    for run in runs {
        for (mean, vector) in means.iter_mut().zip(run) {
            for (target, value) in mean.iter_mut().zip(vector) {
                *target += *value / runs.len() as f32;
            }
        }
    }
    validate_sentinel_vectors(sentinel_count, &means)?;
    Ok(means)
}

fn sentinel_run_passes(
    baseline: &[Vec<f32>],
    observed: &[Vec<f32>],
    gate: f64,
) -> Result<bool, RuntimeError> {
    validate_sentinel_vectors(baseline.len(), observed)?;
    Ok(baseline
        .iter()
        .zip(observed)
        .all(|(left, right)| cosine(left, right).is_ok_and(|similarity| similarity >= gate)))
}

fn cosine(left: &[f32], right: &[f32]) -> Result<f64, RuntimeError> {
    if left.len() != right.len() || left.is_empty() {
        return Err(RuntimeError::protocol(
            "sentinel cosine requires equal non-empty vectors",
        ));
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        if !left.is_finite() || !right.is_finite() {
            return Err(RuntimeError::protocol(
                "sentinel cosine received a non-finite vector",
            ));
        }
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return Err(RuntimeError::protocol(
            "sentinel cosine received a zero-norm vector",
        ));
    }
    let similarity = dot / (left_norm.sqrt() * right_norm.sqrt());
    if !similarity.is_finite() {
        return Err(RuntimeError::protocol(
            "sentinel cosine produced a non-finite similarity",
        ));
    }
    Ok(similarity.clamp(-1.0, 1.0))
}

#[derive(Debug)]
pub(super) struct RuntimeError {
    pub stable: StableError,
    pub message: String,
}

impl RuntimeError {
    pub(super) fn from_parts(stable: StableError, message: impl Into<String>) -> Self {
        Self {
            stable,
            message: message.into(),
        }
    }

    pub(super) fn unavailable(retry_after_ms: u64, provider: &str) -> Self {
        Self::provider_unavailable(retry_after_ms, provider)
    }

    fn invalid_config(message: impl Into<String>) -> Self {
        Self {
            stable: StableError::invalid_request(),
            message: message.into(),
        }
    }

    fn provider_unavailable(retry_after_ms: u64, provider: &str) -> Self {
        Self {
            stable: StableError::provider_unavailable(Some(retry_after_ms)),
            message: format!("provider '{provider}' is unavailable"),
        }
    }

    fn credential_config_invalid(message: impl Into<String>) -> Self {
        Self {
            stable: StableError::credential_config_invalid(),
            message: message.into(),
        }
    }

    fn identity_drift(model: &str) -> Self {
        Self {
            stable: StableError::remote_identity_drift(),
            message: format!("remote identity drift confirmed for model '{model}'"),
        }
    }

    fn protocol(message: impl Into<String>) -> Self {
        Self {
            stable: StableError::provider_protocol_violation(),
            message: message.into(),
        }
    }

    fn store(error: SynapseStoreError) -> Self {
        Self {
            stable: StableError::engine_crashed(Some(100)),
            message: format!("remote runtime store failure: {error}"),
        }
    }
}

impl PartialEq for RuntimeError {
    fn eq(&self, other: &Self) -> bool {
        self.stable == other.stable && self.message == other.message
    }
}

impl Eq for RuntimeError {}

pub(super) struct ProviderRuntime {
    pools: RwLock<HashMap<String, ProviderPool>>,
    pub breaker: CircuitBreaker,
    pub estimator: LatencyEstimator,
    pub credentials: CredentialManager,
    pub continuity: Arc<SentinelContinuityCheck>,
}

impl ProviderRuntime {
    pub(super) fn new(
        breaker_config: BreakerConfig,
        credential_client: Arc<dyn VaultCredentialClient>,
        continuity: Arc<SentinelContinuityCheck>,
    ) -> Result<Self, RuntimeError> {
        Ok(Self {
            pools: RwLock::new(HashMap::new()),
            breaker: CircuitBreaker::new(breaker_config)?,
            estimator: LatencyEstimator::default(),
            credentials: CredentialManager::new(credential_client),
            continuity,
        })
    }

    pub(super) fn register_provider(
        &self,
        provider: &str,
        max_concurrency: usize,
    ) -> Result<(), RuntimeError> {
        let pool = ProviderPool::new(max_concurrency)?;
        self.pools
            .write()
            .expect("provider pools lock poisoned")
            .insert(provider.to_string(), pool);
        Ok(())
    }

    pub(super) fn pool(&self, provider: &str) -> Option<ProviderPool> {
        self.pools
            .read()
            .expect("provider pools lock poisoned")
            .get(provider)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            atomic::{AtomicUsize, Ordering as AtomicOrdering},
            Barrier as ThreadBarrier,
        },
        time::{Duration, Instant},
    };

    use cortexkit_store_types::{Isolation, StorageBackend, StorageDescriptor};
    use reqwest::{Method, Url};
    use tokio::{sync::Barrier, time::sleep};

    use super::*;
    use crate::{
        remote::{
            client::{EndpointSecurity, GatewayHttpClient},
            mock::{MockBehavior, MockProvider},
            openai_compat::{parse_embedding_response, EmbeddingRequest},
            validator::validate_embedding_response,
        },
        store::{CheckpointItem, JobAdmission, JobAttemptClaim},
    };

    fn key() -> BreakerKey {
        BreakerKey {
            provider: "mock-provider".to_string(),
            deployment: "deployment-a".to_string(),
        }
    }

    #[test]
    fn breaker_trips_and_allows_exactly_one_half_open_probe() {
        let breaker = Arc::new(
            CircuitBreaker::new(BreakerConfig {
                failure_threshold: 2,
                cooldown_ms: 100,
            })
            .unwrap(),
        );
        let key = key();
        for now in [0, 1] {
            let lease = breaker.admit(&key, now).unwrap();
            breaker.record_failure(&lease, FailureClass::Transient, now);
        }
        let error = breaker.admit(&key, 50).unwrap_err();
        assert_eq!(error.stable.retry_after_ms, Some(51));

        let start = Arc::new(ThreadBarrier::new(17));
        let probes = Arc::new(Mutex::new(Vec::new()));
        let mut threads = Vec::new();
        for _ in 0..16 {
            let breaker = Arc::clone(&breaker);
            let key = key.clone();
            let start = Arc::clone(&start);
            let probes = Arc::clone(&probes);
            threads.push(std::thread::spawn(move || {
                start.wait();
                if let Ok(probe) = breaker.admit(&key, 101) {
                    probes.lock().unwrap().push(probe);
                }
            }));
        }
        start.wait();
        for thread in threads {
            thread.join().unwrap();
        }
        let mut probes = probes.lock().unwrap();
        assert_eq!(probes.len(), 1);
        let probe = probes.pop().unwrap();
        assert!(probe.is_half_open_probe());
        breaker.record_success(&probe);
        assert_eq!(breaker.state(&key, 101), BreakerStateSnapshot::Closed);
    }

    #[test]
    fn pacing_never_feeds_breaker() {
        let breaker = CircuitBreaker::new(BreakerConfig {
            failure_threshold: 1,
            cooldown_ms: 100,
        })
        .unwrap();
        let key = key();
        let lease = breaker.admit(&key, 0).unwrap();
        breaker.record_failure(
            &lease,
            FailureClass::Pacing {
                retry_after: Some(Duration::from_secs(1)),
            },
            0,
        );
        assert_eq!(breaker.state(&key, 0), BreakerStateSnapshot::Closed);
    }

    #[test]
    fn estimator_uses_bucket_then_merged_fallback_and_max_censor_floor() {
        let estimator = LatencyEstimator::default();
        for (index, duration) in [10, 20, 30, 40, 50, 60, 70, 80].into_iter().enumerate() {
            estimator.observe_latency("p", "embed", 500, duration, index as u64);
        }
        assert_eq!(estimator.estimate_ms("p", "embed", 500, 15_000, 8), 80);
        assert_eq!(estimator.estimate_ms("p", "embed", 5_000, 15_000, 8), 80);
        estimator.observe_censored_timeout("p", "embed", 5_000, 900, 9);
        estimator.observe_censored_timeout("p", "embed", 5_000, 200, 10);
        assert_eq!(estimator.estimate_ms("p", "embed", 5_000, 15_000, 10), 900);
        assert_eq!(
            estimator.estimate_ms("p", "embed", 5_000, 15_000, ESTIMATOR_WINDOW_MS + 10),
            15_000
        );
    }

    #[tokio::test]
    async fn interactive_turnover_precedes_bulk_and_wait_is_one_subbatch() {
        let pool = ProviderPool::new(1).unwrap();
        let first = pool.acquire(RemoteClass::Bulk).await;
        let start = Instant::now();
        let barrier = Arc::new(Barrier::new(3));
        let order = Arc::new(Mutex::new(Vec::new()));

        let interactive_pool = pool.clone();
        let interactive_barrier = Arc::clone(&barrier);
        let interactive_order = Arc::clone(&order);
        let interactive = tokio::spawn(async move {
            interactive_barrier.wait().await;
            let permit = interactive_pool.acquire(RemoteClass::Interactive).await;
            interactive_order.lock().unwrap().push("interactive");
            drop(permit);
        });
        let bulk_pool = pool.clone();
        let bulk_barrier = Arc::clone(&barrier);
        let bulk_order = Arc::clone(&order);
        let bulk = tokio::spawn(async move {
            bulk_barrier.wait().await;
            let permit = bulk_pool.acquire(RemoteClass::Bulk).await;
            bulk_order.lock().unwrap().push("bulk");
            drop(permit);
        });
        barrier.wait().await;
        sleep(Duration::from_millis(20)).await;
        drop(first);
        interactive.await.unwrap();
        bulk.await.unwrap();
        assert_eq!(*order.lock().unwrap(), ["interactive", "bulk"]);
        assert!(start.elapsed() < Duration::from_millis(100));
        assert_eq!(pool.subbatch_tokens(10_000, 20_000, 10_000), 5_000);
    }

    struct ScriptedVault {
        fetches: Mutex<VecDeque<Result<CredentialToken, VaultError>>>,
        min_ttls: Mutex<Vec<u64>>,
        reports: AtomicUsize,
    }

    #[async_trait]
    impl VaultCredentialClient for ScriptedVault {
        async fn fetch(
            &self,
            _logical_handle: &str,
            min_ttl_ms: u64,
        ) -> Result<CredentialToken, VaultError> {
            self.min_ttls.lock().unwrap().push(min_ttl_ms);
            self.fetches.lock().unwrap().pop_front().unwrap()
        }

        async fn report_auth_failure(
            &self,
            _logical_handle: &str,
            _provider_status: u16,
            _record_version: u64,
        ) {
            self.reports.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[tokio::test]
    async fn vault_pause_resume_preserves_committed_checkpoints() {
        let (path, store) = temp_store("vault-resume");
        let generation = store.next_module_generation().unwrap();
        let now = crate::now_ms();
        let record = match store
            .admit_job(
                "request-key",
                "digest",
                "embed_batch",
                generation,
                Some("vault:test"),
                &serde_json::json!({}),
                now,
                10_000,
                10_000,
            )
            .unwrap()
        {
            JobAdmission::Admitted(record) => record,
            JobAdmission::Existing(_) => panic!("unexpected existing job"),
        };
        assert!(matches!(
            store
                .claim_job_attempt(&record.job_id, generation, now + 1)
                .unwrap(),
            JobAttemptClaim::Claimed(_)
        ));
        store
            .commit_job_page(
                &record.job_id,
                0,
                br#"{"items":["a"]}"#,
                &[CheckpointItem {
                    item_id: "a".to_string(),
                    result: br#"[1.0]"#.to_vec(),
                    provider_request_id: Some("upstream-1".to_string()),
                }],
                now + 2,
            )
            .unwrap();

        let vault = Arc::new(ScriptedVault {
            fetches: Mutex::new(VecDeque::from([
                Err(VaultError::NeedsReauth),
                Ok(CredentialToken::new("fresh".to_string(), 100_000, 7)),
            ])),
            min_ttls: Mutex::new(Vec::new()),
            reports: AtomicUsize::new(0),
        });
        let manager = CredentialManager::new(vault.clone() as Arc<dyn VaultCredentialClient>);
        assert!(manager
            .acquire_for_job(
                &store,
                JobCredentialRequest {
                    job_id: &record.job_id,
                    logical_handle: "vault:test",
                    configured_attempt_timeout_ms: 10_000,
                    remaining_deadline_ms: 8_000,
                    now_ms: now + 3,
                    resume_window_ms: 86_400_000,
                },
            )
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            store.get_job(&record.job_id).unwrap().unwrap().state,
            crate::store::JOB_STATE_PAUSED_NEEDS_REAUTH
        );
        assert!(store
            .resume_paused_job(&record.job_id, generation, now + 4, 10_000)
            .unwrap());
        let token = manager
            .acquire_for_job(
                &store,
                JobCredentialRequest {
                    job_id: &record.job_id,
                    logical_handle: "vault:test",
                    configured_attempt_timeout_ms: 10_000,
                    remaining_deadline_ms: 8_000,
                    now_ms: now + 5,
                    resume_window_ms: 86_400_000,
                },
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(token.expose_secret(), "fresh");
        assert_eq!(store.committed_item_ids("digest").unwrap().len(), 1);
        assert_eq!(*vault.min_ttls.lock().unwrap(), [68_000, 68_000]);
        drop(store);
        let _ = std::fs::remove_dir_all(path);
    }

    struct ScriptedEmbedder {
        runs: Mutex<VecDeque<Vec<Vec<f32>>>>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl SentinelEmbedder for ScriptedEmbedder {
        async fn embed_sentinels(
            &self,
            _profile: &SentinelProfile,
            _logical_handle: Option<&str>,
        ) -> Result<Vec<Vec<f32>>, RuntimeError> {
            self.calls.fetch_add(1, AtomicOrdering::Relaxed);
            self.runs
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| RuntimeError::protocol("script exhausted"))
        }
    }

    struct MockHttpEmbedder {
        client: GatewayHttpClient,
        url: Url,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl SentinelEmbedder for MockHttpEmbedder {
        async fn embed_sentinels(
            &self,
            profile: &SentinelProfile,
            _logical_handle: Option<&str>,
        ) -> Result<Vec<Vec<f32>>, RuntimeError> {
            self.calls.fetch_add(1, AtomicOrdering::Relaxed);
            let expected_dimensions = 2;
            let request = self
                .client
                .request(Method::POST, self.url.clone())
                .json(&EmbeddingRequest {
                    model: profile.synapse_model_id.clone(),
                    input: profile.sentinel_texts.clone(),
                    dimensions: Some(expected_dimensions),
                })
                .build()
                .map_err(|error| RuntimeError::protocol(error.to_string()))?;
            let response = self
                .client
                .execute(request, EndpointSecurity::LoopbackAuthNone)
                .await
                .map_err(|error| RuntimeError::protocol(error.to_string()))?;
            if !response.status.is_success() {
                return Err(RuntimeError::provider_unavailable(100, "mock-provider"));
            }
            let parsed = parse_embedding_response(&response.body)
                .map_err(|error| RuntimeError::protocol(error.to_string()))?;
            validate_embedding_response(&parsed, profile.sentinel_texts.len(), expected_dimensions)
                .map(|vectors| {
                    vectors
                        .into_iter()
                        .map(|vector| vector.into_iter().map(|value| value as f32).collect())
                        .collect()
                })
                .map_err(|error| RuntimeError::protocol(error.to_string()))
        }
    }

    #[tokio::test]
    async fn drift_ladder_calibrates_suspects_confirms_and_quarantines() {
        let (path, store) = temp_store("drift-ladder");
        let provider = MockProvider::start().await.unwrap();
        let stable = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let drifted = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        provider.enqueue_all(
            "/embeddings",
            [
                stable.clone(),
                stable.clone(),
                stable.clone(),
                stable.clone(),
                stable,
                drifted.clone(),
                drifted,
            ]
            .into_iter()
            .map(|vectors| MockBehavior::EmbeddingVectors { vectors }),
        );
        let embedder = Arc::new(MockHttpEmbedder {
            client: GatewayHttpClient::new(
                Duration::from_secs(1),
                Duration::from_secs(1),
                1024 * 1024,
            )
            .unwrap(),
            url: provider.url("/embeddings"),
            calls: AtomicUsize::new(0),
        });
        let store = Arc::new(store);
        let continuity = SentinelContinuityCheck::new(
            Arc::clone(&store),
            embedder.clone() as Arc<dyn SentinelEmbedder>,
        );
        let profile = SentinelProfile {
            synapse_model_id: "remote-model".to_string(),
            machine_profile_hash: "machine-a".to_string(),
            remote_profile_hash: "profile-hash".to_string(),
            identity_revision: "r1".to_string(),
            fingerprint: Fingerprint("fingerprint".to_string()),
            numeric_profile_id: NumericProfileId("numeric".to_string()),
            sentinel_texts: vec!["alpha".to_string(), "beta".to_string()],
            drift_gate_min: 0.95,
        };
        continuity
            .calibrate_and_store(&profile, None, 1, 1)
            .await
            .unwrap();
        let error = continuity
            .check("digest", "remote-model", None)
            .await
            .unwrap_err();
        assert!(error.message.contains("remote identity drift"));
        assert_eq!(continuity.state("remote-model"), DriftState::Quarantined);
        assert_eq!(embedder.calls.load(AtomicOrdering::Relaxed), 7);

        let after_restart = SentinelContinuityCheck::empty(store);
        after_restart.register_profile(profile);
        assert!(after_restart
            .check("digest", "remote-model", None)
            .await
            .is_err());
        let _ = std::fs::remove_dir_all(path);
    }

    #[tokio::test]
    async fn calibration_refuses_zero_norm_and_excess_self_noise() {
        let (path, store) = temp_store("calibration-edges");
        let embedder = Arc::new(ScriptedEmbedder {
            runs: Mutex::new(VecDeque::from([vec![vec![0.0, 0.0]]])),
            calls: AtomicUsize::new(0),
        });
        let continuity =
            SentinelContinuityCheck::new(Arc::new(store), embedder as Arc<dyn SentinelEmbedder>);
        let profile = SentinelProfile {
            synapse_model_id: "remote-model".to_string(),
            machine_profile_hash: "machine-a".to_string(),
            remote_profile_hash: "profile-hash".to_string(),
            identity_revision: "r1".to_string(),
            fingerprint: Fingerprint("fingerprint".to_string()),
            numeric_profile_id: NumericProfileId("numeric".to_string()),
            sentinel_texts: vec!["alpha".to_string()],
            drift_gate_min: 0.95,
        };
        let error = continuity.calibrate(&profile, None).await.unwrap_err();
        assert_eq!(error.stable, StableError::provider_protocol_violation());
        let _ = std::fs::remove_dir_all(path);
    }

    #[tokio::test]
    async fn vault_locked_pauses_and_resumes_a_durable_job() {
        let (path, store) = temp_store("vault-locked-resume");
        let generation = store.next_module_generation().unwrap();
        let now = crate::now_ms();
        let record = match store
            .admit_job(
                "locked-request",
                "locked-digest",
                "embed_batch",
                generation,
                Some("vault:test"),
                &serde_json::json!({}),
                now,
                10_000,
                10_000,
            )
            .unwrap()
        {
            JobAdmission::Admitted(record) => record,
            JobAdmission::Existing(_) => panic!("unexpected existing job"),
        };
        assert!(matches!(
            store
                .claim_job_attempt(&record.job_id, generation, now + 1)
                .unwrap(),
            JobAttemptClaim::Claimed(_)
        ));
        let vault = Arc::new(ScriptedVault {
            fetches: Mutex::new(VecDeque::from([Err(VaultError::VaultLocked)])),
            min_ttls: Mutex::new(Vec::new()),
            reports: AtomicUsize::new(0),
        });
        let manager = CredentialManager::new(vault as Arc<dyn VaultCredentialClient>);
        assert!(manager
            .acquire_for_job(
                &store,
                JobCredentialRequest {
                    job_id: &record.job_id,
                    logical_handle: "vault:test",
                    configured_attempt_timeout_ms: 10_000,
                    remaining_deadline_ms: 20_000,
                    now_ms: now + 2,
                    resume_window_ms: 5_000,
                },
            )
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            store.get_job(&record.job_id).unwrap().unwrap().state,
            "paused_needs_reauth"
        );
        assert!(store
            .resume_paused_job(&record.job_id, generation, now + 3, 10_000)
            .unwrap());
        assert_eq!(
            store.get_job(&record.job_id).unwrap().unwrap().state,
            "queued"
        );
        let _ = std::fs::remove_dir_all(path);
    }

    #[tokio::test]
    async fn auth_failure_reporting_excludes_rate_limits_and_server_errors() {
        let vault = Arc::new(ScriptedVault {
            fetches: Mutex::new(VecDeque::new()),
            min_ttls: Mutex::new(Vec::new()),
            reports: AtomicUsize::new(0),
        });
        let manager = CredentialManager::new(vault.clone() as Arc<dyn VaultCredentialClient>);
        manager.report_terminal_auth_failure("handle", 429, 7).await;
        manager.report_terminal_auth_failure("handle", 500, 7).await;
        manager.report_terminal_auth_failure("handle", 401, 7).await;
        manager.report_terminal_auth_failure("handle", 403, 7).await;
        assert_eq!(vault.reports.load(AtomicOrdering::Relaxed), 2);
    }

    fn temp_store(label: &str) -> (std::path::PathBuf, SynapseStore) {
        let path = std::env::temp_dir().join(format!(
            "synapse-runtime-{label}-{}-{}",
            std::process::id(),
            crate::store::JOB_STATE_QUEUED.len() + label.len()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        let descriptor = StorageDescriptor {
            module_id: "synapse-runtime-test".to_string(),
            storage_namespace: "default".to_string(),
            isolation: Isolation::Module,
            backend: StorageBackend::Sqlite {
                path: path.join("store.db").to_string_lossy().to_string(),
            },
        };
        (path, SynapseStore::open(&descriptor).unwrap())
    }
}
