use super::*;

fn db<T>(
    state: &ModuleState,
    f: impl FnOnce(&rusqlite::Connection) -> rusqlite::Result<T>,
) -> Answer<T> {
    state.store.store.with_conn(f).map_err(internal)
}
fn mutate<T>(
    state: &ModuleState,
    f: impl FnOnce(&rusqlite::Transaction<'_>) -> rusqlite::Result<T>,
) -> Answer<T> {
    state.store.store.with_conn_fenced(f).map_err(internal)
}
pub(super) fn directory(state: &ModuleState) -> Answer<PathBuf> {
    let path: String = db(state, |c| {
        c.query_row("PRAGMA database_list", [], |r| r.get(2))
    })?;
    let root = PathBuf::from(path)
        .parent()
        .ok_or_else(|| internal("store has no data directory"))?
        .join("media-audio");
    fs::create_dir_all(&root).map_err(internal)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).map_err(internal)?;
    }
    Ok(root)
}
fn new_id(state: &ModuleState) -> Answer<String> {
    db(state, |c| {
        c.query_row("SELECT lower(hex(randomblob(16)))", [], |r| r.get(0))
    })
}
fn path(state: &ModuleState, id: &str) -> Answer<PathBuf> {
    if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(fault("unknown_object", None));
    }
    Ok(directory(state)?.join(id))
}
fn create(state: &ModuleState, id: &str, bytes: &[u8]) -> Answer<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut f = options.open(path(state, id)?).map_err(internal)?;
    f.write_all(bytes).map_err(internal)?;
    f.sync_all().map_err(internal)
}
fn capacity(state: &ModuleState, bytes: u64) -> Answer<()> {
    let reserved: u64 = db(state, |c| {
        c.query_row(
            "SELECT coalesce(sum(bytes),0) FROM media_objects WHERE state != 'gone'",
            [],
            |r| r.get(0),
        )
    })?;
    if reserved.saturating_add(bytes) > STORAGE_MAX {
        Err(fault("queue_full", None))
    } else {
        Ok(())
    }
}
#[derive(Debug)]
struct Object {
    id: String,
    mime: String,
    bytes: u64,
    sha256: String,
    state: String,
    received: u64,
}
fn object(state: &ModuleState, owner: &str, id: &str, code: &str) -> Answer<Object> {
    db(state, |c| c.query_row("SELECT id,mime,bytes,sha256,state,received FROM media_objects WHERE owner=?1 AND id=?2", params![owner,id], |r| Ok(Object {id:r.get(0)?,mime:r.get(1)?,bytes:r.get(2)?,sha256:r.get(3)?,state:r.get(4)?,received:r.get(5)?})).optional())?.ok_or_else(|| fault(code,None))
}
fn usable(o: &Object) -> Answer<()> {
    match o.state.as_str() {
        "committed" => Ok(()),
        "open" => Err(fault("upload_not_committed", None)),
        _ => Err(fault("unknown_object", None)),
    }
}
fn extend(state: &ModuleState, id: &str, now: u64) -> Answer<()> {
    mutate(state, |t| {
        t.execute(
            "UPDATE media_objects SET retained_ms=max(retained_ms,?2) WHERE id=?1",
            params![id, now.saturating_add(RETENTION_MS)],
        )?;
        Ok(())
    })
}

pub(super) fn upload_begin(state: &ModuleState, owner: &str, p: Value) -> Answer<Value> {
    let p: objects::UploadBeginRequest = parse(p, "params")?;
    if p.mime.is_empty() || p.mime.chars().count() > 256 {
        return Err(invalid("mime"));
    }
    if !objects::sha256_is_valid(&p.sha256) {
        return Err(invalid("sha256"));
    }
    if p.bytes > UPLOAD_MAX {
        return Err(fault("input_too_large", Some("audio")));
    }
    capacity(state, p.bytes)?;
    let id = new_id(state)?;
    create(state, &id, &[])?;
    let result = mutate(state, |t| {
        t.execute("INSERT INTO media_objects(id,owner,mime,bytes,sha256,state,retained_ms) VALUES(?1,?2,?3,?4,?5,'open',?6)",params![id,owner,p.mime,p.bytes,p.sha256,clock_ms(state)+RETENTION_MS])?;
        Ok(())
    });
    if result.is_err() {
        let _ = fs::remove_file(path(state, &id)?);
    }
    result?;
    Ok(json!({"upload_id":id}))
}
pub(super) fn upload_append(state: &ModuleState, owner: &str, p: Value) -> Answer<Value> {
    let p: objects::UploadAppendRequest = parse(p, "params")?;
    let o = object(state, owner, &p.upload_id, "unknown_upload")?;
    match o.state.as_str() {
        "gone" => return Err(fault("upload_gone", None)),
        "committed" => return Err(invalid("upload_id")),
        _ => {}
    }
    let bytes = objects::check_upload_append(&p, o.received).map_err(|e| {
        let mut error = fault(&e.code, None);
        if e.code == "upload_offset_mismatch" {
            error.0["detail"]["received"] = json!(o.received);
        }
        error
    })?;
    if o.received.saturating_add(bytes.len() as u64) > o.bytes {
        return Err(fault("upload_size_mismatch", None));
    }
    let mut f = fs::OpenOptions::new()
        .write(true)
        .open(path(state, &o.id)?)
        .map_err(internal)?;
    // The database is authoritative after an interrupted append. Overwrite an
    // uncommitted file tail rather than accepting bytes SQLite never recorded.
    f.set_len(o.received).map_err(internal)?;
    f.seek(SeekFrom::Start(o.received)).map_err(internal)?;
    f.write_all(&bytes).map_err(internal)?;
    f.sync_all().map_err(internal)?;
    let received = o.received + bytes.len() as u64;
    mutate(state, |t| {
        t.execute(
            "UPDATE media_objects SET received=?2,retained_ms=?3 WHERE id=?1",
            params![o.id, received, clock_ms(state) + RETENTION_MS],
        )?;
        Ok(())
    })?;
    Ok(json!({"received":received}))
}
pub(super) fn upload_status(state: &ModuleState, owner: &str, p: Value) -> Answer<Value> {
    let p: objects::UploadStatusRequest = parse(p, "params")?;
    let o = object(state, owner, &p.upload_id, "unknown_upload")?;
    Ok(json!({"state":o.state,"received":o.received}))
}
pub(super) fn upload_commit(state: &ModuleState, owner: &str, p: Value) -> Answer<Value> {
    let p: objects::UploadCommitRequest = parse(p, "params")?;
    let o = object(state, owner, &p.upload_id, "unknown_upload")?;
    if o.state == "gone" {
        return Err(fault("upload_gone", None));
    }
    if o.state == "open" {
        let bytes = fs::read(path(state, &o.id)?).map_err(internal)?;
        objects::verify_object(&bytes, o.bytes, &o.sha256).map_err(|e| fault(&e.code, None))?;
        mutate(state, |t| {
            t.execute(
                "UPDATE media_objects SET state='committed',retained_ms=?2 WHERE id=?1",
                params![o.id, clock_ms(state) + RETENTION_MS],
            )?;
            Ok(())
        })?;
    }
    Ok(json!({"state":"committed","sha256":o.sha256,"mime":o.mime,"bytes":o.bytes}))
}
pub(super) fn read(state: &ModuleState, owner: &str, p: Value) -> Answer<Value> {
    let p: objects::ReadRequest = parse(p, "params")?;
    let o = object(state, owner, &p.object_id, "unknown_object")?;
    usable(&o)?;
    objects::check_read(&p, o.bytes).map_err(|e| fault(&e.code, e.field.as_deref()))?;
    let output: Option<Vec<u8>> = db(state, |c| {
        c.query_row(
            "SELECT output FROM media_objects WHERE id=?1",
            [&o.id],
            |r| r.get(0),
        )
    })?;
    let bytes = if let Some(output) = output {
        let end = (p.offset + p.length).min(output.len() as u64) as usize;
        output[p.offset as usize..end].to_vec()
    } else {
        let mut f =
            fs::File::open(path(state, &o.id)?).map_err(|_| fault("unknown_object", None))?;
        f.seek(SeekFrom::Start(p.offset)).map_err(internal)?;
        let mut bytes = Vec::new();
        f.take(p.length).read_to_end(&mut bytes).map_err(internal)?;
        bytes
    };
    extend(state, &o.id, clock_ms(state))?;
    mutate(state, |t| {
        t.execute("UPDATE media_jobs SET retained_ms=max(retained_ms,?2) WHERE object_id=?1 OR json_extract(result,'$.output_object.object_id')=?1",params![o.id,clock_ms(state)+RETENTION_MS])?;
        Ok(())
    })?;
    Ok(json!({"data":base64_encode(&bytes)}))
}
pub(super) fn audio_bytes(
    state: &ModuleState,
    owner: &str,
    audio: &objects::MediaValue,
) -> Answer<Vec<u8>> {
    if let Some(data) = &audio.data {
        return STANDARD.decode(data).map_err(|_| invalid("audio"));
    }
    let o = object(
        state,
        owner,
        audio.object_id.as_deref().unwrap_or(""),
        "unknown_object",
    )?;
    usable(&o)?;
    for (field, equal) in [
        ("mime", audio.mime == o.mime),
        ("bytes", audio.bytes == o.bytes),
        ("sha256", audio.sha256 == o.sha256),
    ] {
        if !equal {
            let _ = field;
            return Err(fault("object_mismatch", Some("input.audio")));
        }
    }
    fs::read(path(state, &o.id)?).map_err(|_| fault("unknown_object", None))
}

pub(super) fn binding(p: &Value) -> Value {
    let mut result = json!({});
    // Hash each request member independently so replays can name the first
    // changed field without persisting inline recordings in the backup store.
    for key in ["kind", "input", "model", "decision_id", "deadline_ms"] {
        result[key] = json!(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&p[key]).expect("JSON member"))
        ));
    }
    result
}
pub(super) fn replay(
    state: &ModuleState,
    owner: &str,
    client: &str,
    binding: &Value,
) -> Answer<Option<String>> {
    let row: Option<(String, String)> = db(state, |c| {
        c.query_row(
            "SELECT id,binding FROM media_jobs WHERE owner=?1 AND client_id=?2",
            params![owner, client],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
    })?;
    let Some((id, old)) = row else {
        return Ok(None);
    };
    let old: Value = serde_json::from_str(&old).map_err(internal)?;
    for key in ["kind", "input", "model", "decision_id", "deadline_ms"] {
        if old[key] != binding[key] {
            return Err(fault("client_job_id_reuse", Some(key)));
        }
    }
    Ok(Some(id))
}
pub(super) fn check_lane_capacity(state: &ModuleState, lane: &str) -> Answer<()> {
    let n: u64 = db(state, |c| {
        c.query_row(
            "SELECT count(*) FROM media_jobs WHERE lane=?1 AND state IN ('queued','running')",
            [lane],
            |r| r.get(0),
        )
    })?;
    if n >= LANE_MAX {
        Err(fault("queue_full", None))
    } else {
        Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
pub(super) fn admit(
    state: &ModuleState,
    owner: &str,
    request: &SubmitRequest,
    binding: &Value,
    lane: &str,
    audio: &objects::MediaValue,
    bytes: &[u8],
) -> Answer<String> {
    let new_object = audio.object_id.is_none();
    if new_object {
        capacity(state, audio.bytes)?;
    }
    let id = new_id(state)?;
    let object_id = match &audio.object_id {
        Some(id) => id.clone(),
        None => new_id(state)?,
    };
    if new_object {
        create(state, &object_id, bytes)?;
    }
    let now = clock_ms(state);
    let result = mutate(state, |t| {
        if new_object {
            t.execute("INSERT INTO media_objects(id,owner,mime,bytes,sha256,state,received,retained_ms) VALUES(?1,?2,?3,?4,?5,'committed',?4,?6)",params![object_id,owner,audio.mime,audio.bytes,audio.sha256,now+RETENTION_MS])?;
        }
        t.execute("INSERT INTO media_jobs(id,owner,client_id,binding,lane,object_id,state,accepted_ms,deadline_ms,retained_ms) VALUES(?1,?2,?3,?4,?5,?6,'queued',?7,?8,?9)",params![id,owner,request.client_job_id,binding.to_string(),lane,object_id,now,request.deadline_ms.map(|d|now.saturating_add(d)),now+RETENTION_MS])?;
        t.execute(
            "UPDATE media_objects SET retained_ms=max(retained_ms,?2) WHERE id=?1",
            params![object_id, now + RETENTION_MS],
        )?;
        Ok(())
    });
    if result.is_err() && new_object {
        let _ = fs::remove_file(path(state, &object_id)?);
    }
    result?;
    Ok(id)
}

fn failure(e: Fault) -> Value {
    let mut value = json!({"code":e.0["code"],"class":e.0["detail"]["class"],"message":e.0["message"].as_str().unwrap_or("media failure").chars().take(1024).collect::<String>()});
    let mut detail = e.0["detail"].clone();
    if let Some(retry) = detail
        .as_object_mut()
        .and_then(|m| m.remove("retry_after_ms"))
    {
        value["retry_after_ms"] = retry;
    }
    if let Some(map) = detail.as_object_mut() {
        map.remove("class");
    }
    if detail != json!({}) {
        value["detail"] = detail;
    }
    value
}
pub(super) fn start(state: &ModuleState, id: &str) -> Answer<bool> {
    let _gate = state.runtime.media.gate.lock().expect("media gate");
    expire_deadlines(state, clock_ms(state))?;
    mutate(state, |t| {
        Ok(t.execute(
            "UPDATE media_jobs SET state='running' WHERE id=?1 AND state='queued'",
            [id],
        )? == 1)
    })
}
pub(super) fn finish(state: &ModuleState, id: &str, result: Answer<Value>) -> Answer<()> {
    expire_deadlines(state, clock_ms(state))?;
    let (mut status, mut output, mut error) = match result {
        Ok(v) => ("done", Some(v), None),
        Err(e) => ("failed", None, Some(failure(e))),
    };
    mutate(state, |t| {
        let pending: bool = t.query_row(
            "SELECT state IN ('queued','running') FROM media_jobs WHERE id=?1",
            [id],
            |r| r.get(0),
        )?;
        if !pending {
            return Ok(());
        }
        if let Some(observation) = output
            .as_mut()
            .and_then(|v| v.as_object_mut())
            .and_then(|m| m.remove("_observation"))
        {
            let mut metadata = observation.clone();
            for part in ["encoder_output", "decode_trace"] {
                let bytes = STANDARD
                    .decode(observation[part].as_str().unwrap_or(""))
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                for (page, data) in bytes.chunks(PAGE).enumerate() {
                    t.execute("INSERT INTO media_observation_pages(job_id,part,page,data) VALUES(?1,?2,?3,?4)",params![id,part,page as u64,data])?;
                }
                metadata[part] = json!({"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes)),"pages":bytes.len().div_ceil(PAGE)});
            }
            attach_certify_observation(output.as_mut().expect("output"), Some(metadata));
        }
        let now = clock_ms(state);
        if let Some(value) = &output {
            let bytes = serde_json::to_vec(value).expect("JSON output");
            if bytes.len() > cortexkit_role_media::MESSAGE_MAX {
                let reserved: u64 = t.query_row(
                    "SELECT coalesce(sum(bytes),0) FROM media_objects WHERE state!='gone'",
                    [],
                    |r| r.get(0),
                )?;
                if reserved.saturating_add(bytes.len() as u64) > STORAGE_MAX {
                    status = "failed";
                    output = None;
                    error = Some(failure(fault("queue_full", None)));
                    t.execute("DELETE FROM media_observation_pages WHERE job_id=?1", [id])?;
                } else {
                    let object: String =
                        t.query_row("SELECT lower(hex(randomblob(16)))", [], |r| r.get(0))?;
                    let owner: String =
                        t.query_row("SELECT owner FROM media_jobs WHERE id=?1", [id], |r| {
                            r.get(0)
                        })?;
                    let sha = format!("{:x}", Sha256::digest(&bytes));
                    t.execute("INSERT INTO media_objects(id,owner,mime,bytes,sha256,state,received,retained_ms,output) VALUES(?1,?2,'application/json',?3,?4,'committed',?3,?5,?6)",params![object,owner,bytes.len() as u64,sha,now+RETENTION_MS,bytes])?;
                    output = Some(
                        json!({"output_object":{"mime":"application/json","bytes":bytes.len(),"sha256":sha,"object_id":object}}),
                    );
                }
            }
        }
        t.execute(
            "UPDATE media_jobs SET state=?2,result=?3,failure=?4,retained_ms=?5 WHERE id=?1",
            params![
                id,
                status,
                output.map(|v| v.to_string()),
                error.map(|v| v.to_string()),
                now + RETENTION_MS
            ],
        )?;
        t.execute("UPDATE media_objects SET retained_ms=max(retained_ms,?2) WHERE id=(SELECT object_id FROM media_jobs WHERE id=?1)",params![id,now+RETENTION_MS])?;
        Ok(())
    })
}
fn expire_deadlines(state: &ModuleState, now: u64) -> Answer<()> {
    let error = failure(fault("deadline_exceeded", None)).to_string();
    mutate(state, |t| {
        t.execute("UPDATE media_objects SET retained_ms=max(retained_ms,?1) WHERE id IN (SELECT object_id FROM media_jobs WHERE state IN ('queued','running') AND deadline_ms <= ?2)",params![now+RETENTION_MS,now])?;
        t.execute("UPDATE media_jobs SET state='failed',failure=?1,retained_ms=?2 WHERE state IN ('queued','running') AND deadline_ms<=?3",params![error,now+RETENTION_MS,now])?;
        Ok(())
    })
}
pub(super) fn job_answer(
    state: &ModuleState,
    owner: &str,
    method: &str,
    p: Value,
) -> Answer<Value> {
    let id = string(&p, "job_id")?;
    expire_deadlines(state, clock_ms(state))?;
    let row: Option<(String, Option<String>, Option<String>, String)> = db(state, |c| {
        c.query_row("SELECT state,result,failure,object_id FROM media_jobs WHERE owner=?1 AND id=?2 AND expired=0",params![owner,id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()
    })?;
    let Some((status, result, error, object)) = row else {
        return Err(fault("unknown_job", None));
    };
    if method == "media.cancel" {
        if status == "queued" || status == "running" {
            mutate(state, |t| {
                t.execute("UPDATE media_jobs SET state='cancelled',retained_ms=?2 WHERE id=?1 AND state IN ('queued','running')",params![id,clock_ms(state)+RETENTION_MS])?;
                Ok(())
            })?;
            extend(state, &object, clock_ms(state))?;
            return Ok(json!({"status":"cancelled"}));
        }
        return Ok(json!({"status":status}));
    }
    if method == "media.status" {
        let mut v = json!({"status":status});
        if let Some(error) = error {
            v["failure"] = serde_json::from_str(&error).map_err(internal)?;
        }
        return Ok(v);
    }
    if status != "done" {
        let mut e = fault("job_not_done", None);
        e.0["detail"]["class"] = json!(if matches!(status.as_str(), "queued" | "running") {
            "transient"
        } else {
            "permanent"
        });
        e.0["detail"]["status"] = json!(status);
        return Err(e);
    }
    mutate(state, |t| {
        t.execute(
            "UPDATE media_jobs SET retained_ms=?2 WHERE id=?1",
            params![id, clock_ms(state) + RETENTION_MS],
        )?;
        Ok(())
    })?;
    extend(state, &object, clock_ms(state))?;
    let result: Value = serde_json::from_str(&result.unwrap_or_default()).map_err(internal)?;
    if let Some(id) = result["output_object"]["object_id"].as_str() {
        extend(state, id, clock_ms(state))?;
    }
    Ok(result)
}

pub(super) fn observation(state: &ModuleState, owner: &str, p: Value) -> Answer<Value> {
    if !state.runtime.certify_observation {
        return Err(fault("certify_observation_disabled", None));
    }
    let id = string(&p, "job_id")?;
    let part = string(&p, "part")?;
    if !matches!(part, "encoder_output" | "decode_trace") {
        return Err(invalid("part"));
    }
    let page = p["page"].as_u64().ok_or_else(|| invalid("page"))?;
    let known: bool = db(state, |c| {
        c.query_row(
            "SELECT EXISTS(SELECT 1 FROM media_jobs WHERE owner=?1 AND id=?2 AND expired=0)",
            params![owner, id],
            |r| r.get(0),
        )
    })?;
    if !known {
        return Err(fault("unknown_job", None));
    }
    let data: Option<Vec<u8>> = db(state, |c| {
        c.query_row(
            "SELECT data FROM media_observation_pages WHERE job_id=?1 AND part=?2 AND page=?3",
            params![id, part, page],
            |r| r.get(0),
        )
        .optional()
    })?;
    let data = data.ok_or_else(|| invalid("page"))?;
    Ok(json!({"data":base64_encode(&data)}))
}

pub(super) fn sweep(state: &ModuleState, now: u64) -> Answer<()> {
    expire_deadlines(state, now)?;
    let ids: Vec<String> = db(state, |c| {
        let mut q=c.prepare("SELECT id FROM media_objects WHERE state!='gone' AND retained_ms<=?1 AND NOT EXISTS(SELECT 1 FROM media_jobs WHERE object_id=media_objects.id AND state IN ('queued','running'))")?;
        let ids = q.query_map([now], |r| r.get(0))?.collect();
        ids
    })?;
    for id in ids {
        match fs::remove_file(path(state, &id)?) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(internal(e)),
        };
        mutate(state, |t| {
            t.execute(
                "UPDATE media_objects SET state='gone',output=NULL WHERE id=?1",
                [id],
            )?;
            Ok(())
        })?;
    }
    mutate(state, |t| {
        t.execute("DELETE FROM media_observation_pages WHERE job_id IN (SELECT id FROM media_jobs WHERE state NOT IN ('queued','running') AND retained_ms<=?1)",[now])?;
        t.execute("UPDATE media_jobs SET result=NULL,expired=1 WHERE state NOT IN ('queued','running') AND retained_ms<=?1",[now])?;
        Ok(())
    })
}
pub(super) fn recover(state: &ModuleState) -> Answer<()> {
    let root = directory(state)?;
    let now = clock_ms(state);
    let error = failure(fault("module_restarted", None)).to_string();
    mutate(state, |t| {
        t.execute("UPDATE media_objects SET retained_ms=max(retained_ms,?1) WHERE id IN (SELECT object_id FROM media_jobs WHERE state IN ('queued','running'))",[now+RETENTION_MS])?;
        t.execute("UPDATE media_jobs SET state='failed',failure=?1,retained_ms=?2 WHERE state IN ('queued','running')",params![error,now+RETENTION_MS])?;
        t.execute(
            "UPDATE media_objects SET state='gone' WHERE state='open'",
            [],
        )?;
        Ok(())
    })?;
    let ids: Vec<String> = db(state, |c| {
        let mut q = c.prepare("SELECT id FROM media_objects WHERE state='committed'")?;
        let ids = q.query_map([], |r| r.get(0))?.collect();
        ids
    })?;
    let keep: BTreeSet<_> = ids.into_iter().collect();
    for entry in fs::read_dir(root).map_err(internal)? {
        let entry = entry.map_err(internal)?;
        if entry.file_type().map_err(internal)?.is_file()
            && !keep.contains(&entry.file_name().to_string_lossy().to_string())
        {
            fs::remove_file(entry.path()).map_err(internal)?;
        }
    }
    sweep(state, now)
}

pub(super) fn job_audio(state: &ModuleState, id: &str) -> Answer<Vec<u8>> {
    let object: String = db(state, |c| {
        c.query_row("SELECT object_id FROM media_jobs WHERE id=?1", [id], |r| {
            r.get(0)
        })
    })?;
    fs::read(path(state, &object)?).map_err(internal)
}
