CREATE TABLE event_outbox (
    event_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    published_at INTEGER
);
CREATE INDEX event_outbox_pending ON event_outbox(created_at) WHERE published_at IS NULL;
CREATE INDEX event_outbox_published ON event_outbox(published_at) WHERE published_at IS NOT NULL;
CREATE TABLE findings_state (
    id INTEGER PRIMARY KEY CHECK(id=0),
    digest TEXT NOT NULL,
    findings_generation INTEGER NOT NULL DEFAULT 0
);
INSERT INTO findings_state(id,digest) VALUES(0,'{"worst_severity":"ok","findings":[]}');

-- Media terminal transitions and download state changes announce
-- facts through triggers so every mutation path enqueues in the fact's transaction.
-- Generate each UUID once, then reuse it in the row key and its JSON body.
CREATE TRIGGER event_media_terminal AFTER UPDATE OF state ON media_jobs
WHEN OLD.state IS NOT NEW.state AND NEW.state IN ('done','failed','cancelled')
BEGIN
    INSERT INTO event_outbox(event_id,subject,body,created_at)
    SELECT event_id,'event.synapse.job_page.v1',
        json_object('event_id',event_id,'job_id',NEW.id,'page_seq',0,'job_state',NEW.state),
        CAST((julianday('now')-2440587.5)*86400000 AS INTEGER)
    FROM (
        WITH event AS MATERIALIZED (
            SELECT lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' ||
                substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',1+(random() & 3),1) ||
                substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6))) AS event_id
        ) SELECT event_id FROM event
    );
END;

CREATE TRIGGER event_download_created AFTER INSERT ON jobs WHEN NEW.kind='models.download'
BEGIN
    INSERT INTO event_outbox(event_id,subject,body,created_at)
    SELECT event_id,'event.synapse.download_state.v1',
        json_object('event_id',event_id,'download_id',NEW.job_id,'state',NEW.state),NEW.updated_ms
    FROM (
        WITH event AS MATERIALIZED (
            SELECT lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' ||
                substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',1+(random() & 3),1) ||
                substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6))) AS event_id
        ) SELECT event_id FROM event
    );
END;

CREATE TRIGGER event_download_changed AFTER UPDATE OF state ON jobs
WHEN NEW.kind='models.download' AND OLD.state IS NOT NEW.state
BEGIN
    INSERT INTO event_outbox(event_id,subject,body,created_at)
    SELECT event_id,'event.synapse.download_state.v1',
        json_object('event_id',event_id,'download_id',NEW.job_id,'state',NEW.state),NEW.updated_ms
    FROM (
        WITH event AS MATERIALIZED (
            SELECT lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' ||
                substr(lower(hex(randomblob(2))),2) || '-' || substr('89ab',1+(random() & 3),1) ||
                substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6))) AS event_id
        ) SELECT event_id FROM event
    );
END;
