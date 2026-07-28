//! Canonical decode identity, quarantine key, and runtime-config digest.
//!
//! Three identities are kept deliberately separate so cache invalidation,
//! certification, substitution, and wire provenance use the narrowest correct
//! identity:
//!
//! - [`DecodeIdentity`] — the canonical lane identity. Production decode is
//!   `engine=owned-metal-decode`, `task=generate`, `lane=decode`,
//!   `worker=supervised`, `risk_class=abort_capable`. Embedding remains
//!   `engine=owned-metal`, `risk_class=abort_safe` and is untouched here.
//! - `decode_fingerprint` — identifies the canonical-token-ID-to-generated-token-ID
//!   function. Scheduler settings never rotate it.
//! - `runtime_config_digest` — SHA-256 over the canonical worker runtime manifest
//!   ([`RuntimeManifest`]). Its scheduler inputs are exactly the five runtime-effective
//!   fields (resolution r2 #1); the cancellation-latency bound and other workload
//!   or evidence values never enter it.
//!
//! A supervised worker is scoped to one [`QuarantineKey`] and hosts no other key.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Worker protocol ID. Legacy llama frames carrying raw grammar are never sent
/// to a worker speaking this protocol.
pub const WORKER_PROTOCOL_ID: &str = "owned-metal-decode-worker-v1";

/// Constraint encoding carried over the worker boundary. Raw schema or grammar
/// never crosses the boundary; only the compiled token-ID constraint does.
pub const CONSTRAINT_ENCODING_ID: &str = "token-id-json-constraint-v1";

/// Canonical engine name for production decode.
pub const ENGINE: &str = "owned-metal-decode";
/// Canonical task.
pub const TASK: &str = "generate";
/// Canonical lane.
pub const LANE: &str = "decode";
/// Canonical worker mode.
pub const WORKER: &str = "supervised";
/// Canonical risk class. Embedding remains `abort_safe`; decode is `abort_capable`.
pub const RISK_CLASS: &str = "abort_capable";

/// The canonical owned-decode lane identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeIdentity {
    pub engine: &'static str,
    pub task: &'static str,
    pub lane: &'static str,
    pub worker: &'static str,
    pub risk_class: &'static str,
}

impl DecodeIdentity {
    /// The single canonical production-decode identity.
    #[must_use]
    pub const fn production() -> Self {
        Self {
            engine: ENGINE,
            task: TASK,
            lane: LANE,
            worker: WORKER,
            risk_class: RISK_CLASS,
        }
    }
}

impl Default for DecodeIdentity {
    fn default() -> Self {
        Self::production()
    }
}

/// The quarantine key a supervised worker is scoped to. Each worker hosts
/// exactly one key; crashes charge only that key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuarantineKey {
    pub machine_profile_hash: String,
    pub decode_fingerprint: String,
    pub runtime_config_digest: String,
}

impl QuarantineKey {
    #[must_use]
    pub fn new(
        machine_profile_hash: impl Into<String>,
        decode_fingerprint: impl Into<String>,
        runtime_config_digest: impl Into<String>,
    ) -> Self {
        Self {
            machine_profile_hash: machine_profile_hash.into(),
            decode_fingerprint: decode_fingerprint.into(),
            runtime_config_digest: runtime_config_digest.into(),
        }
    }

    /// A stable rendering used as the persistence key for crash-budget and
    /// quarantine state.
    #[must_use]
    pub fn storage_id(&self) -> String {
        format!(
            "{}|{}|{}",
            self.machine_profile_hash, self.decode_fingerprint, self.runtime_config_digest
        )
    }
}

/// The five runtime-effective scheduler fields. Exactly these enter
/// `runtime_config_digest`; none enters `decode_fingerprint`. The
/// cancellation-latency bound is a derived quantity and is deliberately absent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerRuntimeRecord {
    /// The single committed production N, one of `{8, 16, 32}`. N=1 is prohibited.
    pub production_n: u32,
    pub yield_policy_revision: String,
    pub decode_aging_window_ms: u64,
    pub decode_weight: u32,
    pub progress_protocol_revision: String,
}

impl SchedulerRuntimeRecord {
    /// Validate the committed N. Exactly one production N is selected from
    /// `{8,16,32}`; N=1 and every other value are rejected (resolution r2 #5).
    pub fn validate(&self) -> Result<(), &'static str> {
        match self.production_n {
            8 | 16 | 32 => Ok(()),
            _ => Err("production_n must be exactly one of {8, 16, 32}"),
        }
    }
}

/// The canonical worker runtime manifest. `runtime_config_digest` is SHA-256
/// over exactly the fields enumerated in [`RuntimeManifest::digest_fields`], in
/// that order. Fields documented as non-runtime (cancellation-latency bound,
/// workload, evidence) are intentionally not present here so they cannot enter
/// the digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeManifest {
    pub worker_revision: String,
    pub protocol_revision: String,
    pub metallib_revision: String,
    /// Baseline K is one; later chain-K requires N divisible by K.
    pub chain_k: u32,
    /// Baseline batched verification is disabled.
    pub batched_verification: bool,
    /// Version 1 permits exactly one resident generation.
    pub resident_limit: u32,
    /// Attention-KV reservation input (context bucket).
    pub attention_kv_bucket: u32,
    /// LFM2 convolution-cache reservation input; zero for families without a
    /// convolution rolling cache.
    pub lfm2_conv_cache_units: u32,
    pub context_manifest_revision: String,
    pub crash_policy_revision: String,
    pub quarantine_duration_ms: u64,
    pub scheduler: SchedulerRuntimeRecord,
}

impl RuntimeManifest {
    /// The ordered `(name, value)` pairs the digest covers. Order is part of the
    /// identity: reordering rotates the digest.
    #[must_use]
    pub fn digest_fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("worker_revision", self.worker_revision.clone()),
            ("protocol_revision", self.protocol_revision.clone()),
            ("metallib_revision", self.metallib_revision.clone()),
            ("chain_k", self.chain_k.to_string()),
            (
                "batched_verification",
                self.batched_verification.to_string(),
            ),
            ("resident_limit", self.resident_limit.to_string()),
            ("attention_kv_bucket", self.attention_kv_bucket.to_string()),
            (
                "lfm2_conv_cache_units",
                self.lfm2_conv_cache_units.to_string(),
            ),
            (
                "context_manifest_revision",
                self.context_manifest_revision.clone(),
            ),
            ("crash_policy_revision", self.crash_policy_revision.clone()),
            (
                "quarantine_duration_ms",
                self.quarantine_duration_ms.to_string(),
            ),
            // The five runtime-effective scheduler fields, inlined so the digest
            // covers exactly them and nothing else from the scheduler record.
            ("production_n", self.scheduler.production_n.to_string()),
            (
                "yield_policy_revision",
                self.scheduler.yield_policy_revision.clone(),
            ),
            (
                "decode_aging_window_ms",
                self.scheduler.decode_aging_window_ms.to_string(),
            ),
            ("decode_weight", self.scheduler.decode_weight.to_string()),
            (
                "progress_protocol_revision",
                self.scheduler.progress_protocol_revision.clone(),
            ),
        ]
    }

    /// Compute `runtime_config_digest` as hex-encoded SHA-256 over the canonical
    /// field encoding.
    #[must_use]
    pub fn runtime_config_digest(&self) -> String {
        let mut hasher = Sha256::new();
        for (name, value) in self.digest_fields() {
            // Length-prefixed framing keeps field boundaries unambiguous so a
            // value can never bleed into the next field's bytes.
            hasher.update(name.as_bytes());
            hasher.update(b"=");
            hasher.update(value.len().to_le_bytes());
            hasher.update(value.as_bytes());
            hasher.update(b";");
        }
        hex_encode(&hasher.finalize())
    }
}

/// Lowercase-hex encode a digest without pulling in the `hex` crate.
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest(n: u32) -> RuntimeManifest {
        RuntimeManifest {
            worker_revision: "worker-v1".into(),
            protocol_revision: WORKER_PROTOCOL_ID.into(),
            metallib_revision: "metallib-r1".into(),
            chain_k: 1,
            batched_verification: false,
            resident_limit: 1,
            attention_kv_bucket: 2048,
            lfm2_conv_cache_units: 0,
            context_manifest_revision: "decode-context-buckets-v1".into(),
            crash_policy_revision: "crash-policy-v1".into(),
            quarantine_duration_ms: 60_000,
            scheduler: SchedulerRuntimeRecord {
                production_n: n,
                yield_policy_revision: "yield-v1".into(),
                decode_aging_window_ms: 50,
                decode_weight: 4,
                progress_protocol_revision: "progress-v1".into(),
            },
        }
    }

    #[test]
    fn canonical_identity_is_the_documented_decode_lane() {
        let identity = DecodeIdentity::production();
        assert_eq!(identity.engine, "owned-metal-decode");
        assert_eq!(identity.task, "generate");
        assert_eq!(identity.lane, "decode");
        assert_eq!(identity.worker, "supervised");
        assert_eq!(identity.risk_class, "abort_capable");
    }

    #[test]
    fn production_n_is_restricted_to_the_committed_set() {
        assert!(sample_manifest(8).scheduler.validate().is_ok());
        assert!(sample_manifest(16).scheduler.validate().is_ok());
        assert!(sample_manifest(32).scheduler.validate().is_ok());
        assert!(sample_manifest(1).scheduler.validate().is_err());
        assert!(sample_manifest(64).scheduler.validate().is_err());
    }

    #[test]
    fn runtime_digest_is_stable_and_scheduler_sensitive() {
        let a = sample_manifest(16);
        let b = sample_manifest(16);
        assert_eq!(a.runtime_config_digest(), b.runtime_config_digest());
        // Changing a runtime-effective scheduler field rotates the digest.
        let c = sample_manifest(32);
        assert_ne!(a.runtime_config_digest(), c.runtime_config_digest());
        // The digest is 64 hex chars (SHA-256).
        assert_eq!(a.runtime_config_digest().len(), 64);
    }

    #[test]
    fn quarantine_key_storage_id_is_stable() {
        let key = QuarantineKey::new("profile", "fp", "rt");
        assert_eq!(key.storage_id(), "profile|fp|rt");
    }
}
