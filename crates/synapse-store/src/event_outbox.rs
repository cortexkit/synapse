use rusqlite::{params, OptionalExtension};

use crate::{SynapseStore, SynapseStoreError};

pub(crate) const MIGRATION: &str = include_str!("event_outbox.sql");
const RETENTION_MS: u64 = 7 * 24 * 60 * 60 * 1000;

pub(crate) fn enqueue_job_page(
    tx: &rusqlite::Transaction<'_>,
    job_id: &str,
    page_seq: u32,
    state: &str,
    now: u64,
) -> rusqlite::Result<()> {
    tx.execute(
        "INSERT INTO event_outbox(event_id,subject,body,created_at) \
         SELECT event_id,'event.synapse.job_page.v1', \
         json_object('event_id',event_id,'job_id',?1,'page_seq',?2,'job_state',?3),?4 \
         FROM (WITH event AS MATERIALIZED (SELECT lower(hex(randomblob(4))) || '-' || \
         lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || \
         substr('89ab',1+(random() & 3),1) || substr(lower(hex(randomblob(2))),2) || '-' || \
         lower(hex(randomblob(6))) AS event_id) SELECT event_id FROM event)",
        params![job_id, page_seq, state, now],
    )?;
    Ok(())
}

#[derive(Debug)]
pub struct Event {
    pub event_id: String,
    pub subject: String,
    pub body: String,
}

impl SynapseStore {
    pub fn next_event(&self) -> Result<Option<Event>, SynapseStoreError> {
        Ok(self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT event_id, subject, body FROM event_outbox WHERE published_at IS NULL \
                 ORDER BY created_at, rowid LIMIT 1",
                [],
                |row| {
                    Ok(Event {
                        event_id: row.get(0)?,
                        subject: row.get(1)?,
                        body: row.get(2)?,
                    })
                },
            )
            .optional()
        })?)
    }

    pub fn mark_event_published(&self, id: &str, now: u64) -> Result<(), SynapseStoreError> {
        self.store.with_conn_fenced(|tx| {
            tx.execute("UPDATE event_outbox SET published_at=?2 WHERE event_id=?1 AND published_at IS NULL", params![id, now])?;
            Ok(())
        })?;
        Ok(())
    }

    pub fn prune_events(&self, now: u64) -> Result<(), SynapseStoreError> {
        if let Some(cutoff) = now.checked_sub(RETENTION_MS) {
            self.store.with_conn_fenced(|tx| {
                tx.execute("DELETE FROM event_outbox WHERE published_at < ?1", [cutoff])?;
                Ok(())
            })?;
        }
        Ok(())
    }

    pub fn record_findings_digest(
        &self,
        digest: &str,
        worst_severity: &str,
        now: u64,
    ) -> Result<bool, SynapseStoreError> {
        Ok(self.store.with_conn_fenced(|tx| {
            let changed = tx.execute(
                "UPDATE findings_state SET digest=?1, findings_generation=findings_generation+1 \
                 WHERE id=0 AND digest IS NOT ?1", [digest],
            )?;
            if changed == 0 { return Ok(false); }
            let generation: u64 = tx.query_row("SELECT findings_generation FROM findings_state WHERE id=0", [], |r| r.get(0))?;
            tx.execute(
                "INSERT INTO event_outbox(event_id,subject,body,created_at) \
                 SELECT event_id,'event.synapse.findings_changed.v1', \
                 json_object('event_id',event_id,'generation',?1,'worst_severity',?2),?3 \
                 FROM (WITH event AS MATERIALIZED (SELECT lower(hex(randomblob(4))) || '-' || \
                 lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || \
                 substr('89ab',1+(random() & 3),1) || substr(lower(hex(randomblob(2))),2) || '-' || \
                 lower(hex(randomblob(6))) AS event_id) SELECT event_id FROM event)",
                params![generation,worst_severity,now],
            )?;
            Ok(true)
        })?)
    }
}
