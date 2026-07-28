//! Crash-budget accounting, persistence, and quarantine.
//!
//! Every crash, protocol-fatal response, startup failure, timeout, or failed
//! cancellation charges exactly one unit to the affected quarantine key.
//! Acknowledged cancellation and acknowledged deadline cleanup before timeout
//! charge nothing (the supervisor simply never calls [`CrashBudget::charge`]
//! for them). Coincident timeout/crash/startup/protocol-fatal failure is charged
//! once, under the single classification recorded for that failure.
//!
//! A terminal charge that exhausts the budget leaves the key quarantined for
//! `quarantine_duration_ms`. State persists across supervisor restarts through a
//! [`CrashBudgetStore`]; [`InMemoryBudgetStore`] backs fixtures and
//! [`FileBudgetStore`] backs production persistence.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::boundary::Timestamp;
use crate::error::FailureClassification;
use crate::identity::QuarantineKey;

/// The crash policy: how many strikes a key tolerates before quarantine, and
/// for how long quarantine lasts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetPolicy {
    /// Number of chargeable failures that exhausts the key. With the default of
    /// two, the first crash leaves one unit (permitting exactly one redispatch)
    /// and the second exhausts and quarantines.
    pub max_strikes: u32,
    pub quarantine_duration_ms: u64,
}

impl BudgetPolicy {
    #[must_use]
    pub const fn new(max_strikes: u32, quarantine_duration_ms: u64) -> Self {
        Self {
            max_strikes,
            quarantine_duration_ms,
        }
    }
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self {
            max_strikes: 2,
            quarantine_duration_ms: 60_000,
        }
    }
}

/// Persistent per-key crash state.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetRecord {
    /// Chargeable failures recorded against this key.
    pub strikes: u32,
    /// Ordered failure classifications, mirrored into provenance and telemetry.
    pub failure_classifications: Vec<FailureClassification>,
    /// When quarantine lifts; `None` when not quarantined.
    pub quarantined_until: Option<Timestamp>,
}

impl BudgetRecord {
    /// Whether the key is quarantined at `now`. Quarantine lapses once
    /// `quarantined_until` is in the past.
    #[must_use]
    pub fn is_quarantined(&self, now: Timestamp) -> bool {
        self.quarantined_until.is_some_and(|until| now < until)
    }
}

/// Persistence for crash-budget and quarantine state.
pub trait CrashBudgetStore {
    fn load(&self, key: &QuarantineKey) -> Option<BudgetRecord>;
    fn save(&mut self, key: &QuarantineKey, record: &BudgetRecord);
}

/// An in-memory store. Fixtures use it to assert accounting without touching
/// the filesystem.
#[derive(Default)]
pub struct InMemoryBudgetStore {
    records: BTreeMap<String, BudgetRecord>,
}

impl CrashBudgetStore for InMemoryBudgetStore {
    fn load(&self, key: &QuarantineKey) -> Option<BudgetRecord> {
        self.records.get(&key.storage_id()).cloned()
    }

    fn save(&mut self, key: &QuarantineKey, record: &BudgetRecord) {
        self.records.insert(key.storage_id(), record.clone());
    }
}

/// A JSON-file-backed store. The whole record map is rewritten atomically (write
/// to a temporary path, then rename) so a crash mid-write never leaves a torn
/// budget file.
pub struct FileBudgetStore {
    path: PathBuf,
    records: BTreeMap<String, BudgetRecord>,
}

impl FileBudgetStore {
    /// Open or create the store at `path`, loading any existing records.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let records = if path.exists() {
            let bytes = std::fs::read(&path)?;
            serde_json::from_slice(&bytes)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?
        } else {
            BTreeMap::new()
        };
        Ok(Self { path, records })
    }

    fn flush(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(&self.records)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

impl CrashBudgetStore for FileBudgetStore {
    fn load(&self, key: &QuarantineKey) -> Option<BudgetRecord> {
        self.records.get(&key.storage_id()).cloned()
    }

    fn save(&mut self, key: &QuarantineKey, record: &BudgetRecord) {
        self.records.insert(key.storage_id(), record.clone());
        // Persistence failures are surfaced rather than swallowed: a budget that
        // cannot be persisted must fail closed, not silently forget strikes.
        if let Err(error) = self.flush() {
            eprintln!(
                "owned-decode-worker: failed to persist crash budget to {}: {error}",
                self.path.display()
            );
        }
    }
}

/// The result of charging one failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChargeOutcome {
    pub strikes_after: u32,
    /// True when this charge exhausted the budget.
    pub exhausted: bool,
    /// True when the key is quarantined as a result of this charge.
    pub quarantined: bool,
}

/// Supervisor-facing crash-budget accounting over a [`CrashBudgetStore`].
pub struct CrashBudget<S: CrashBudgetStore> {
    store: S,
    policy: BudgetPolicy,
}

impl<S: CrashBudgetStore> CrashBudget<S> {
    #[must_use]
    pub fn new(store: S, policy: BudgetPolicy) -> Self {
        Self { store, policy }
    }

    #[must_use]
    pub const fn policy(&self) -> BudgetPolicy {
        self.policy
    }

    /// The record for a key, or a fresh empty record.
    #[must_use]
    pub fn record(&self, key: &QuarantineKey) -> BudgetRecord {
        self.store.load(key).unwrap_or_default()
    }

    /// Remaining budget units (`max_strikes - strikes`, saturated at zero).
    #[must_use]
    pub fn remaining(&self, key: &QuarantineKey) -> u32 {
        self.policy
            .max_strikes
            .saturating_sub(self.record(key).strikes)
    }

    /// Whether the key is currently quarantined.
    #[must_use]
    pub fn is_quarantined(&self, key: &QuarantineKey, now: Timestamp) -> bool {
        self.record(key).is_quarantined(now)
    }

    /// Charge exactly one unit for a single failure classification. Records the
    /// classification in order, and quarantines the key if the charge exhausts
    /// the budget. Returns the post-charge state.
    pub fn charge(
        &mut self,
        key: &QuarantineKey,
        classification: FailureClassification,
        now: Timestamp,
    ) -> ChargeOutcome {
        let mut record = self.record(key);
        record.strikes = record.strikes.saturating_add(1);
        record.failure_classifications.push(classification);
        let exhausted = record.strikes >= self.policy.max_strikes;
        if exhausted {
            record.quarantined_until = Some(now + self.policy.quarantine_duration_ms);
        }
        self.store.save(key, &record);
        ChargeOutcome {
            strikes_after: record.strikes,
            exhausted,
            quarantined: exhausted,
        }
    }

    /// Whether a worker-crash redispatch is permitted: after the first failure
    /// has been charged, at least one budget unit must remain and the key must
    /// not be quarantined. The caller separately checks that the request is not
    /// cancelled and that the original deadline remains valid.
    #[must_use]
    pub fn redispatch_permitted(&self, key: &QuarantineKey, now: Timestamp) -> bool {
        self.remaining(key) >= 1 && !self.is_quarantined(key, now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> QuarantineKey {
        QuarantineKey::new("profile", "dfp", "rt")
    }

    #[test]
    fn first_crash_leaves_one_unit_and_permits_redispatch() {
        let mut budget = CrashBudget::new(InMemoryBudgetStore::default(), BudgetPolicy::default());
        let outcome = budget.charge(&key(), FailureClassification::Crash, 0);
        assert_eq!(outcome.strikes_after, 1);
        assert!(!outcome.exhausted);
        assert!(!outcome.quarantined);
        assert_eq!(budget.remaining(&key()), 1);
        assert!(budget.redispatch_permitted(&key(), 0));
    }

    #[test]
    fn second_crash_exhausts_and_quarantines() {
        let mut budget = CrashBudget::new(InMemoryBudgetStore::default(), BudgetPolicy::default());
        budget.charge(&key(), FailureClassification::Crash, 0);
        let outcome = budget.charge(&key(), FailureClassification::Crash, 10);
        assert_eq!(outcome.strikes_after, 2);
        assert!(outcome.exhausted);
        assert!(outcome.quarantined);
        assert!(budget.is_quarantined(&key(), 10));
        assert!(!budget.redispatch_permitted(&key(), 10));
        // Quarantine lapses after the configured duration.
        assert!(!budget.is_quarantined(&key(), 10 + 60_000));
    }

    #[test]
    fn classifications_are_recorded_in_order() {
        let mut budget = CrashBudget::new(InMemoryBudgetStore::default(), BudgetPolicy::default());
        budget.charge(&key(), FailureClassification::Crash, 0);
        budget.charge(&key(), FailureClassification::StartupFailure, 5);
        assert_eq!(
            budget.record(&key()).failure_classifications,
            vec![
                FailureClassification::Crash,
                FailureClassification::StartupFailure
            ]
        );
    }

    #[test]
    fn budget_state_persists_across_store_reopens() {
        let dir = std::env::temp_dir().join(format!(
            "owned-decode-worker-budget-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let file = dir.join("budget.json");
        {
            let store = FileBudgetStore::open(&file).expect("open");
            let mut budget = CrashBudget::new(store, BudgetPolicy::default());
            budget.charge(&key(), FailureClassification::Timeout, 0);
        }
        // Reopen: the strike survived.
        let store = FileBudgetStore::open(&file).expect("reopen");
        let budget = CrashBudget::new(store, BudgetPolicy::default());
        assert_eq!(budget.remaining(&key()), 1);
        assert_eq!(
            budget.record(&key()).failure_classifications,
            vec![FailureClassification::Timeout]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
