//! The unlink phase of `models.remove`.
//!
//! `models.remove` first commits the deletion of its install, member and
//! self-check rows, then calls into this module while still holding the
//! `catalog_disk` lock. Unlinking cannot be rolled back, so it runs only after
//! that commit: a crash or an unlink error here leaves unreferenced bytes on
//! disk for `cache.gc`, never database rows naming files that are gone.

use super::*;
use store::CatalogBlobCache;

/// Unlinks each of `digests` that no install, download, registration or pin
/// still references, then reclaims owned Metal packages left without roots.
/// Returns the bytes actually unlinked. A failed root check or unlink is
/// logged and skipped: the removal has already committed, and the remaining
/// bytes are unrooted, so ordinary cache garbage collection can retry them.
pub(super) fn reclaim_unrooted_blobs(state: &ModuleState, digests: &[String]) -> u64 {
    let cache = CatalogCache(&state.model_cache);
    let mut freed = 0u64;
    for digest in digests {
        let referenced = state
            .store
            .sqlite()
            .with_conn_fenced(|tx| catalog_blob_referenced(tx, digest));
        match referenced {
            Ok(false) => {}
            Ok(true) => continue,
            Err(error) => {
                tracing::warn!(%digest, %error, "could not check catalog blob roots after removal");
                continue;
            }
        }
        match cache.is_pinned(digest) {
            Ok(false) => {}
            Ok(true) => continue,
            Err(error) => {
                tracing::warn!(%digest, %error, "could not read catalog blob pins after removal");
                continue;
            }
        }
        match cache.delete_blob(digest) {
            Ok(bytes) => freed = freed.saturating_add(bytes),
            Err(error) => {
                tracing::warn!(%digest, %error, "could not unlink unrooted catalog blob after removal");
            }
        }
    }
    match state
        .store
        .sqlite()
        .with_conn_fenced(|tx| reclaim_unrooted_owned_packages(tx, &state.model_cache))
    {
        Ok(bytes) => freed = freed.saturating_add(bytes),
        Err(error) => {
            tracing::warn!(%error, "could not reclaim unrooted owned packages after removal");
        }
    }
    freed
}
