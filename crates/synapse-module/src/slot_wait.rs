//! Waiting on a `Notify` for shared state to change without losing the wake-up.
//!
//! `Notify::notify_waiters` wakes only the `Notified` futures that already
//! exist; it leaves no permit for a future created afterwards. A waiter that
//! reads the state, sees it unsettled and only then subscribes can therefore
//! miss a change published between the read and the subscription, and sleep
//! until its deadline although the state has settled. Subscribing first and
//! reading the state second closes that gap.

use tokio::sync::Notify;

/// What a state read found.
pub(crate) enum SlotCheck<T> {
    /// The state has settled; the wait ends with this value.
    Settled(T),
    /// Still unsettled: wait for the next notification.
    Pending,
    /// The state moved somewhere `notify` no longer reports on, for example a
    /// replaced slot with its own `Notify`; the caller reads again at once.
    Changed,
}

/// Why a wait ended without a settled value.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SlotWait {
    /// A notification arrived, or the state changed owner; read again.
    Recheck,
    /// The deadline passed with the state still unsettled.
    TimedOut,
}

/// Subscribes to `notify`, then runs `check`. Returns the settled value, or
/// waits for a notification (including one sent while `check` ran) until
/// `deadline`.
pub(crate) async fn check_then_wait<T>(
    notify: &Notify,
    deadline: tokio::time::Instant,
    check: impl FnOnce() -> SlotCheck<T>,
) -> Result<T, SlotWait> {
    let notified = notify.notified();
    tokio::pin!(notified);
    // An enabled `Notified` receives `notify_waiters` even before its first poll.
    notified.as_mut().enable();
    match check() {
        SlotCheck::Settled(value) => return Ok(value),
        SlotCheck::Changed => return Err(SlotWait::Recheck),
        SlotCheck::Pending => {}
    }
    match tokio::time::timeout_at(deadline, notified).await {
        Ok(()) => Err(SlotWait::Recheck),
        Err(_) => Err(SlotWait::TimedOut),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Another thread publishes the settled state and notifies right after the
    /// waiter's read found it unsettled. The waiter must wake for that
    /// notification instead of sleeping until its deadline.
    #[tokio::test(start_paused = true)]
    async fn a_notification_sent_after_the_read_still_wakes_the_waiter() {
        let notify = Notify::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let outcome = check_then_wait(&notify, deadline, || {
            notify.notify_waiters();
            SlotCheck::<()>::Pending
        })
        .await;
        assert_eq!(outcome, Err(SlotWait::Recheck));
        assert!(
            tokio::time::Instant::now() < deadline,
            "the waiter slept until its deadline"
        );
    }
}
