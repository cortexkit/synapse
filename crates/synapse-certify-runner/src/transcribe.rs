//! Pinned media jobs, sealed observation paging and measured speech release gates.
use crate::{
    catalog::{bounded, Client, Pin},
    refuse,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path, time::Duration};
use synapse_certify::{transcribe::*, Catalog, Result};
use synapse_parity::{
    canonical::{sha256_file, sha256_hex},
    speech::{self, SpeechFixture, SpeechObservation, SpeechProfile},
};
use tokio::time::Instant;

mod shards;
#[cfg(test)]
mod tests;

fn err(e: impl std::fmt::Display) -> synapse_certify::Error {
    refuse(e.to_string())
}
fn number(value: &Value, name: &str) -> Result<u64> {
    value[name]
        .as_u64()
        .ok_or_else(|| refuse(format!("missing integer field: {name}")))
}
fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value[name]
        .as_str()
        .ok_or_else(|| refuse(format!("missing string field: {name}")))
}
async fn call(
    client: &mut impl Client,
    method: &str,
    params: Value,
    deadline: Instant,
) -> Result<Value> {
    let value = request_with_retry(client, method, params, deadline).await?;
    if value.get("error").is_some() {
        return Err(refuse(format!("{method} refused: {value}")));
    }
    Ok(value)
}

async fn request_with_retry(
    client: &mut impl Client,
    method: &str,
    params: Value,
    deadline: Instant,
) -> Result<Value> {
    const MAX_RETRIES: u32 = 3;
    let mut retries = 0;
    loop {
        let response = bounded(client, method, params.clone(), deadline).await?;
        let error = &response["error"];
        let code = error["code"].as_str();
        let detail = &error["detail"];
        // A rerun that starts consumes its upload, even on failure. Retry only
        // refusals that occur before the candidate takes ownership of the object.
        if method == "certify.transcribe_observe_rerun"
            && !matches!(code, Some("queue_full" | "model_loading" | "lane_paused"))
        {
            return Ok(response);
        }
        let cold = method == "media.submit" && code == Some("model_loading");
        let hint = detail["retry_after_ms"].as_u64();
        let delay = if cold {
            // Loading may take longer than a few retries, but shares the row deadline.
            hint.or_else(|| error["retry_after_ms"].as_u64())
                .unwrap_or(100)
        } else if let Some(hint) =
            hint.filter(|_| detail["class"] == "transient" && retries < MAX_RETRIES)
        {
            retries += 1;
            hint
        } else {
            return Ok(response);
        };
        let delay = Duration::from_millis(delay.max(1));
        if delay >= deadline.saturating_duration_since(Instant::now()) {
            return Err(refuse(format!(
                "{method} retry would exceed runner deadline: {response}"
            )));
        }
        tokio::time::timeout_at(deadline, tokio::time::sleep(delay))
            .await
            .map_err(|_| refuse("runner deadline exceeded"))?;
    }
}

async fn pause(deadline: Instant) -> Result<()> {
    tokio::time::timeout_at(deadline, tokio::time::sleep(Duration::from_millis(100)))
        .await
        .map_err(err)
}

pub(crate) async fn read_part(
    client: &mut impl Client,
    job: &str,
    part: &str,
    descriptor: &Value,
    deadline: Instant,
) -> Result<(Vec<u8>, Part)> {
    let bytes = number(descriptor, "bytes")?;
    let pages = number(descriptor, "pages")?;
    let sha256 = text(descriptor, "sha256")?.to_string();
    // The largest encoder has 8192 rows. A bounded descriptor prevents a corrupt
    // candidate from requesting arbitrary memory or an unbounded page loop.
    if bytes > 256 * 1024 * 1024 || pages != bytes.div_ceil(524288) {
        return Err(refuse("observation part has invalid size or page count"));
    }
    let mut raw = Vec::with_capacity(bytes as usize);
    for page in 0..pages {
        let response = call(
            client,
            "certify.transcribe_observation",
            json!({"job_id":job,"part":part,"page":page}),
            deadline,
        )
        .await?;
        let data = STANDARD.decode(text(&response, "data")?).map_err(err)?;
        let expected = (bytes - raw.len() as u64).min(524288) as usize;
        if data.len() != expected {
            return Err(refuse("observation page length mismatch"));
        }
        raw.extend(data);
    }
    let reassembled_sha256 = sha256_hex(&raw);
    if raw.len() as u64 != bytes || reassembled_sha256 != sha256 {
        return Err(refuse("observation bytes or sha256 mismatch"));
    }
    let seal = Part {
        bytes,
        pages,
        sha256,
        reassembled_bytes: raw.len() as u64,
        reassembled_sha256,
    };
    Ok((raw, seal))
}

fn triples(raw: &str) -> Result<Vec<speech::Triple>> {
    let bytes = STANDARD.decode(raw).map_err(err)?;
    if bytes.len() % 12 != 0 {
        return Err(refuse("malformed transcript frame"));
    }
    Ok(bytes
        .as_chunks::<12>()
        .0
        .iter()
        .map(|b| {
            std::array::from_fn(|i| i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
        })
        .collect())
}

async fn upload(client: &mut impl Client, wav: &[u8], deadline: Instant) -> Result<Value> {
    let digest = sha256_hex(wav);
    let begun = call(
        client,
        "media.upload_begin",
        json!({"mime":"audio/wav","bytes":wav.len(),"sha256":digest}),
        deadline,
    )
    .await?;
    let id = text(&begun, "upload_id")?;
    for (index, chunk) in wav.chunks(2 * 1024 * 1024).enumerate() {
        let got = call(
            client,
            "media.upload_append",
            json!({"upload_id":id,"offset":index*2*1024*1024,"data":STANDARD.encode(chunk)}),
            deadline,
        )
        .await?;
        if number(&got, "received")? != ((index * 2 * 1024 * 1024) + chunk.len()) as u64 {
            return Err(refuse("upload offset mismatch"));
        }
    }
    let committed = call(
        client,
        "media.upload_commit",
        json!({"upload_id":id}),
        deadline,
    )
    .await?;
    if committed["state"] != "committed" || committed["sha256"] != digest {
        return Err(refuse("upload commit mismatch"));
    }
    Ok(json!({"object_id":id,"mime":"audio/wav","bytes":wav.len(),"sha256":digest}))
}

async fn submit(
    client: &mut impl Client,
    pin: &Pin,
    fixture: &str,
    audio: Value,
    deadline: Instant,
) -> Result<Value> {
    let params = json!({"client_job_id":format!("certify-{fixture}"),"kind":"transcribe","model":pin.lane_id,"input":{"audio":audio},"deadline_ms":3600000});
    request_with_retry(client, "media.submit", params, deadline).await
}

async fn terminal_status(client: &mut impl Client, job: &str, deadline: Instant) -> Result<Value> {
    loop {
        let status = call(client, "media.status", json!({"job_id":job}), deadline).await?;
        match status["status"].as_str() {
            Some("queued" | "running") => pause(deadline).await?,
            _ => return Ok(status),
        }
    }
}

async fn result(
    client: &mut impl Client,
    job: &str,
    object_budget: u64,
    deadline: Instant,
) -> Result<Value> {
    let status = terminal_status(client, job, deadline).await?;
    if status["status"] != "done" {
        return Err(refuse(format!("transcribe job failed: {status}")));
    }
    let value = call(client, "media.result", json!({"job_id":job}), deadline).await?;
    if value.get("output_object").is_none() {
        return Ok(value);
    }
    let object = &value["output_object"];
    let mut raw = Vec::new();
    let bytes = number(object, "bytes")?;
    if bytes > object_budget {
        return Err(refuse("media result exceeds fixture's storage budget"));
    }
    while (raw.len() as u64) < bytes {
        let length = (bytes - raw.len() as u64).min(2 * 1024 * 1024);
        let page = call(
            client,
            "media.read",
            json!({"object_id":object["object_id"],"offset":raw.len(),"length":length}),
            deadline,
        )
        .await?;
        let data = STANDARD.decode(text(&page, "data")?).map_err(err)?;
        if data.len() as u64 != length {
            return Err(refuse("media result page length mismatch"));
        }
        raw.extend(data);
    }
    if sha256_hex(&raw) != text(object, "sha256")? {
        return Err(refuse("media result digest mismatch"));
    }
    serde_json::from_slice(&raw).map_err(err)
}

pub(crate) async fn observation(
    client: &mut impl Client,
    pin: &Pin,
    job: &str,
    response: &Value,
    deadline: Instant,
) -> Result<(SpeechObservation, Part, Part)> {
    let observed = &response["observation"];
    if observed["lane_id"] != pin.lane_id || observed["fingerprint"] != pin.fingerprint {
        return Err(refuse(
            "transcribe observation lane_id or fingerprint differs from pin",
        ));
    }
    let (encoder, encoder_seal) = read_part(
        client,
        job,
        "encoder_output",
        &observed["encoder_output"],
        deadline,
    )
    .await?;
    let (trace, trace_seal) = read_part(
        client,
        job,
        "decode_trace",
        &observed["decode_trace"],
        deadline,
    )
    .await?;
    if encoder.len() % 4 != 0 {
        return Err(refuse("encoder observation is not f32le"));
    }
    Ok((
        SpeechObservation {
            valid_encoder_frames: Some(number(observed, "valid_encoder_frames")?),
            engine_started: true,
            encoder_output: Some(
                encoder
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
            ),
            trace: speech::decode_trace(&trace).map_err(err)?,
            triples: triples(text(observed, "triples")?)?,
            text: text(response, "text")?.to_string(),
            refusal: None,
        },
        encoder_seal,
        trace_seal,
    ))
}

pub(crate) async fn self_check(
    client: &mut impl Client,
    pin: &Pin,
    deadline: Instant,
) -> Result<SelfCheck> {
    let listed = call(client, "models.list", json!({}), deadline).await?;
    let lane = listed["models"]
        .as_array()
        .and_then(|rows| rows.iter().find(|r| r["model_id"] == pin.lane_id))
        .ok_or_else(|| refuse("loaded catalog lane missing from models.list"))?;
    if lane["self_check"]["state"] != "passed" || lane["fingerprints"] != json!([pin.fingerprint]) {
        return Err(refuse(format!(
            "catalog lane check or pin mismatch: {lane}"
        )));
    }
    Ok(SelfCheck {
        state: "passed".into(),
        fingerprints: vec![pin.fingerprint.clone()],
    })
}

pub(crate) async fn measurements(
    client: &mut impl Client,
    job: &str,
    backend: &str,
    deadline: Instant,
) -> Result<Measurements> {
    let surface = "certify.transcribe_measurements";
    let value = call(client, surface, json!({"job_id":job}), deadline)
        .await
        .map_err(|e| refuse(format!("{surface} measurement surface unavailable: {e}")))?;
    let mut fields = vec![
        "self_check_wall_ms",
        "longest_progress_interval_ms",
        "ane_shapes",
    ];
    if matches!(backend, "cuda" | "vulkan") {
        fields.push("peak_device_memory_bytes");
    }
    if backend == "vulkan" {
        fields.push("peak_storage_buffer_bytes");
    }
    for field in fields {
        if value.get(field).is_none_or(Value::is_null) {
            return Err(refuse(format!("missing {surface}.{field}")));
        }
    }
    serde_json::from_value(value).map_err(|e| refuse(format!("invalid {surface}: {e}")))
}

#[derive(Clone, Copy)]
enum CounterSource<'a> {
    Worker,
    InProcess(&'a str),
}

async fn counters(
    client: &mut impl Client,
    source: CounterSource<'_>,
    deadline: Instant,
) -> Result<(u64, u64)> {
    let value = call(client, "certify.observations", json!({}), deadline).await?;
    if value["available"] != true {
        return Err(refuse("certify.observations unavailable"));
    }
    let sent = match source {
        CounterSource::InProcess(lane) => {
            number(&value["in_process_transcribe_engine_invocations"], lane)?
        }
        CounterSource::Worker => value["worker_requests"]
            .as_object()
            .ok_or_else(|| refuse("worker request counters missing"))?
            .values()
            .try_fold(0u64, |sum, v| sum.checked_add(v.as_u64()?).or(None))
            .ok_or_else(|| refuse("invalid worker counters"))?,
    };
    Ok((sent, number(&value, "admitted_count")?))
}

async fn refusal(
    client: &mut impl Client,
    fixture: &SpeechFixture,
    response: &Value,
    profile: &SpeechProfile,
    source: CounterSource<'_>,
    before: (u64, u64),
    deadline: Instant,
) -> Result<ExpectedRefusal> {
    // A duration limit discovered from the audio itself is a terminal job failure.
    let status;
    let fault = if let Some(job) = response["job_id"].as_str() {
        status = terminal_status(client, job, deadline).await?;
        if status["status"] != "failed" {
            return Err(refuse(format!(
                "over-limit transcribe job did not fail: {status}"
            )));
        }
        &status["failure"]
    } else {
        &response["error"]
    };
    let after = counters(client, source, deadline).await?;
    let detail = &fault["detail"];
    let details = [
        number(detail, "duration_ms")?,
        number(detail, "max_duration_ms")?,
        number(detail, "valid_encoder_frames")?,
        number(detail, "max_encoder_frames")?,
    ];
    let code = text(fault, "code")?;
    let engine_calls = after.0.checked_sub(before.0).ok_or_else(|| match source {
        CounterSource::Worker => refuse("worker counter regressed"),
        CounterSource::InProcess(_) => refuse("in-process transcribe engine counter regressed"),
    })?;
    let shape_admissions = after
        .1
        .checked_sub(before.1)
        .ok_or_else(|| refuse("shape counter regressed"))?;
    speech::grade_applicability(
        profile,
        "input_too_large",
        fixture.duration_ms,
        fixture.sample_count,
        None,
        Some(&speech::SpeechRefusal {
            code: code.into(),
            details,
        }),
        engine_calls != 0 || shape_admissions != 0,
    )
    // Name what was observed: a bare refusal gives the operator nothing to
    // compare against the fixture's sealed duration and the lane's limit.
    .map_err(|e| {
        refuse(format!(
            "{e}: fixture {} ({} ms, {} samples) against limit {} ms; refusal {code} {details:?}; \
             engine calls {engine_calls}, shape admissions {shape_admissions}",
            fixture.id, fixture.duration_ms, fixture.sample_count, profile.max_duration_ms
        ))
    })?;
    if detail["field"] != "input.audio" || detail["limit"] != "max_duration_ms" {
        return Err(refuse("duration refusal field or limit mismatch"));
    }
    Ok(ExpectedRefusal {
        id: fixture.id.clone(),
        code: code.into(),
        field: "input.audio".into(),
        limit: "max_duration_ms".into(),
        duration_ms: details[0],
        max_duration_ms: details[1],
        valid_encoder_frames: details[2],
        max_encoder_frames: details[3],
        worker_requests: match source {
            CounterSource::Worker => engine_calls,
            CounterSource::InProcess(_) => 0,
        },
        shape_admissions,
    })
}

struct FixturePaths<'a> {
    parity: &'a Path,
    mirror: &'a Path,
}

#[derive(Default)]
struct Collected {
    graded: Vec<GradedFixture>,
    refusals: Vec<ExpectedRefusal>,
    reruns: Vec<String>,
    self_check: Option<SelfCheck>,
    measurements: Option<Measurements>,
    ran_maximum: bool,
}

impl Collected {
    fn merge_measurements(&mut self, measured: Measurements) {
        if let Some(all) = &mut self.measurements {
            all.longest_progress_interval_ms = all
                .longest_progress_interval_ms
                .max(measured.longest_progress_interval_ms);
            all.peak_device_memory_bytes = all
                .peak_device_memory_bytes
                .max(measured.peak_device_memory_bytes);
            all.peak_storage_buffer_bytes = all
                .peak_storage_buffer_bytes
                .max(measured.peak_storage_buffer_bytes);
            for shape in measured.ane_shapes {
                if !all
                    .ane_shapes
                    .iter()
                    .any(|s| s.encoder_frames == shape.encoder_frames)
                {
                    all.ane_shapes.push(shape);
                }
            }
        } else {
            // Keep the first load's self-check time rather than replacing it
            // with later loads' timings when combining shards.
            self.measurements = Some(measured);
        }
    }

    fn merge(&mut self, shard: Self) {
        self.graded.extend(shard.graded);
        self.refusals.extend(shard.refusals);
        self.reruns.extend(shard.reruns);
        self.self_check = self.self_check.take().or(shard.self_check);
        self.ran_maximum |= shard.ran_maximum;
        if let Some(measured) = shard.measurements {
            self.merge_measurements(measured);
        }
    }
}

async fn collect(
    client: &mut impl Client,
    pin: &Pin,
    profile: &SpeechProfile,
    inputs: &[shards::Input<'_>],
    paths: FixturePaths<'_>,
    sharded: bool,
    deadline: Instant,
) -> Result<Collected> {
    let mut collected = Collected::default();
    let backend = pin
        .lane_id
        .rsplit('-')
        .next()
        .ok_or_else(|| refuse("lane backend missing"))?;
    let counter_source = if backend == "metal" {
        CounterSource::InProcess(&pin.lane_id)
    } else {
        CounterSource::Worker
    };
    for input in inputs {
        let fixture = input.fixture;
        let wav = std::fs::read(&input.path).map_err(err)?;
        if wav.len() as u64 != input.audio_bytes || sha256_hex(&wav) != input.digest {
            return Err(refuse("cached audio size or digest mismatch"));
        }
        let audio = upload(client, &wav, deadline).await?;
        let above = fixture.duration_ms > profile.max_duration_ms;
        let before = if above {
            Some(counters(client, counter_source, deadline).await?)
        } else {
            None
        };
        let submitted = submit(client, pin, &fixture.id, audio, deadline).await?;
        if above {
            collected.refusals.push(
                refusal(
                    client,
                    fixture,
                    &submitted,
                    profile,
                    counter_source,
                    before.unwrap(),
                    deadline,
                )
                .await?,
            );
            continue;
        }
        if submitted.get("error").is_some() {
            return Err(refuse(format!("in-limit fixture refused: {submitted}")));
        }
        let job = text(&submitted, "job_id")?;
        let response = result(client, job, input.result_bytes, deadline).await?;
        let (observed, encoder_output, decode_trace) =
            observation(client, pin, job, &response, deadline).await?;
        if collected.self_check.is_none() {
            collected.self_check = Some(self_check(client, pin, deadline).await?);
        }
        // Score numerical gates one fixture at a time to avoid retaining gigabytes
        // of encoder tensors. Split WER is aggregated after every fixture ran.
        let mut numerical = fixture.clone();
        numerical.split = None;
        numerical.ground_truth = None;
        let mut observations = BTreeMap::new();
        observations.insert(fixture.id.clone(), observed);
        let report = speech::evaluate_transcribe(
            profile,
            &[numerical],
            &observations,
            paths.parity,
            paths.mirror,
            "input_too_large",
        )
        .map_err(err)?;
        let observed = &observations[&fixture.id];
        let diagnostics = &report.diagnostics[&fixture.id];
        if fixture.encoder.is_some() {
            let rerun_audio = upload(client, &wav, deadline).await?;
            let off = call(
                client,
                "certify.transcribe_observe_rerun",
                json!({"job_id":job,"audio":rerun_audio}),
                deadline,
            )
            .await?;
            if off["observe"] != false
                || number(&off, "valid_encoder_frames")? != observed.valid_encoder_frames.unwrap()
                || triples(text(&off, "triples")?)? != observed.triples
            {
                return Err(refuse("observe on/off emitted triples or frames differ"));
            }
            collected.reruns.push(fixture.id.clone());
        }
        let is_maximum =
            fixture.id.starts_with("ladder-") && fixture.duration_ms == profile.max_duration_ms;
        collected.ran_maximum |= is_maximum;
        if sharded || is_maximum {
            let measured = measurements(client, job, backend, deadline).await?;
            if measured.self_check_wall_ms >= 2000 {
                return Err(refuse("speech self-check time gate failed"));
            }
            collected.merge_measurements(measured);
        }
        collected.graded.push(GradedFixture {
            id: fixture.id.clone(),
            lane_id: pin.lane_id.clone(),
            fingerprint: pin.fingerprint.clone(),
            valid_encoder_frames: observed.valid_encoder_frames.unwrap(),
            text: observed.text.clone(),
            encoder_output,
            decode_trace,
            integrity_passed: true,
            trace_passed: diagnostics.trace_error.is_none(),
            margin_exempt: report.margin_exempt.contains(&fixture.id),
            encoder_min_cosine: diagnostics.encoder_min_cosine,
            encoder_error: diagnostics.encoder_error.clone(),
            trace_error: diagnostics.trace_error.clone(),
        });
    }
    if collected.self_check.is_none() {
        return Err(refuse("self-check not observed in speech shard"));
    }
    Ok(collected)
}

fn verify_shard_binaries(
    layout: &crate::assets::Layout,
    artifacts: &[synapse_certify::Artifact],
) -> Result<()> {
    for artifact in artifacts {
        let path = match artifact.role.as_str() {
            "ck-synapse" => layout.candidate.clone(),
            role if role.starts_with("cuda-runtime:") => {
                layout.root.join(role.trim_start_matches("cuda-runtime:"))
            }
            _ => layout
                .worker
                .clone()
                .ok_or_else(|| refuse("candidate worker alias missing"))?,
        };
        if sha256_file(&path).map_err(err)? != artifact.sha256 {
            return Err(refuse(format!(
                "speech shard binary digest differs: {}",
                artifact.role
            )));
        }
    }
    Ok(())
}

async fn collect_shards(
    options: &crate::live::Options,
    layout: &crate::assets::Layout,
    row: &str,
    profile: &mut SpeechProfile,
    inputs: &[shards::Input<'_>],
    executed_artifacts: &[synapse_certify::Artifact],
) -> Result<(Pin, Collected)> {
    let shards = shards::partition(inputs)?;
    let is_sharded = shards.len() > 1;
    let mirror = crate::mirror::Mirror::start(&options.mirror).await?;
    let deadline = Instant::now() + crate::catalog::RUN_BUDGET;
    let mut first_pin: Option<Pin> = None;
    let mut all = Collected::default();
    let parity = options.checkout.join("bench/parity");
    for (index, shard) in shards.into_iter().enumerate() {
        let root = if index == 0 {
            layout.root.clone()
        } else {
            let root = layout.root.join(format!("s{index}"));
            std::fs::create_dir(&root).map_err(err)?;
            root
        };
        verify_shard_binaries(layout, executed_artifacts)?;
        let mut client = tokio::time::timeout_at(
            deadline,
            crate::live::start_in(layout, &root, &mirror.endpoint),
        )
        .await
        .map_err(|_| refuse("runner deadline exceeded starting speech shard"))??;
        let outcome = async {
            let pin = crate::catalog::install(&mut client, row, speech::MODEL, deadline).await?;
            if pin.task != "transcribe" {
                return Err(refuse("catalog task differs from transcribe"));
            }
            let describe = call(&mut client, "role.describe", json!({}), deadline).await?;
            let engine = describe["engines"]
                .as_array()
                .and_then(|rows| rows.iter().find(|r| r["model"] == pin.lane_id))
                .ok_or_else(|| refuse("role.describe omitted pinned speech engine"))?;
            let certified = number(engine, "max_duration_ms")?;
            if let Some(first) = &first_pin {
                if first.lane_id != pin.lane_id
                    || first.fingerprint != pin.fingerprint
                    || certified != profile.max_duration_ms
                {
                    return Err(refuse("speech shard catalog pin or duration differs"));
                }
            } else {
                if certified == 0 || certified > profile.max_duration_ms {
                    return Err(refuse("role.describe exceeds profile duration"));
                }
                profile.max_duration_ms = certified;
            }
            let collected = collect(
                &mut client,
                &pin,
                profile,
                &inputs[shard],
                FixturePaths {
                    parity: &parity,
                    mirror: &options.mirror,
                },
                is_sharded,
                deadline,
            )
            .await?;
            Ok((pin, collected))
        }
        .await;
        client.shutdown().await?;
        let (pin, collected) = outcome?;
        first_pin.get_or_insert(pin);
        all.merge(collected);
    }
    if !all.ran_maximum {
        return Err(refuse("maximum-duration ladder was not executed"));
    }
    Ok((
        first_pin.ok_or_else(|| refuse("no speech shards executed"))?,
        all,
    ))
}

pub(crate) async fn run(
    options: &crate::live::Options,
    source: &str,
    row: &str,
    floor: Option<&Value>,
    layout: &crate::assets::Layout,
) -> Result<Value> {
    let required = required_machine(row);
    let machine = crate::live::machine(row, floor)
        .map_err(|e| refuse(format!("real hardware pass required on {required}: {e}")))?;
    let fixtures = fixtures(&options.checkout)?;
    let profile_id = format!("{}.{}", speech::MODEL, crate::live::lane(row));
    let manifest = synapse_certify::self_check::manifest()?;
    let mut profile = manifest.profiles[&profile_id]
        .transcribe
        .clone()
        .ok_or_else(|| refuse("transcribe profile missing"))?;
    let audio_root = layout.root.join("transcribe-audio");
    // Windows Python installs as `python`; `python3` there is only the Store's
    // install prompt, which exits without running anything.
    let python = if cfg!(windows) { "python" } else { "python3" };
    let prepared = synapse_core::without_launch_nonce_tokio(tokio::process::Command::new(python))
        .arg(
            options
                .checkout
                .join("crates/synapse-certify-runner/scripts/prepare-transcribe-audio.py"),
        )
        .arg("--repo")
        .arg(&options.checkout)
        .arg("--cache")
        .arg(&options.mirror)
        .arg("--output")
        .arg(&audio_root)
        .kill_on_drop(true)
        .status()
        .await
        .map_err(err)?;
    if !prepared.success() {
        return Err(refuse(
            "speech audio preparation failed; install the pinned reference environment",
        ));
    }
    let inputs = shards::inputs(&audio_root, &fixtures, profile.max_duration_ms)?;
    let executed_artifacts = layout
        .artifacts
        .iter()
        .map(|a| {
            Ok(synapse_certify::Artifact {
                role: a.role.clone(),
                file: a.file.clone(),
                sha256: synapse_certify::digest(&options.assets, &a.file)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let outcome = async {
        let (pin, collected) = collect_shards(
            options,
            layout,
            row,
            &mut profile,
            &inputs,
            &executed_artifacts,
        )
        .await?;
        let certified = profile.max_duration_ms;
        let graded_fixtures = collected.graded;
        let by_id: BTreeMap<_, _> = graded_fixtures.iter().map(|g| (g.id.as_str(), g)).collect();
        let split_wer = speech::grade_wer(fixtures.iter().filter_map(|f| {
            Some((
                f.split.as_deref()?,
                f.ground_truth.as_deref()?,
                f.text.as_str(),
                by_id.get(f.id.as_str())?.text.as_str(),
            ))
        }))
        .map_err(err)?;
        let catalog = Catalog {
            lanes: vec![(
                speech::MODEL.into(),
                synapse_certify::row_backend(row).unwrap().into(),
                crate::live::lane(row).into(),
                pin.fingerprint.clone(),
            )],
            tasks: BTreeMap::from([(speech::MODEL.into(), "transcribe".into())]),
        };
        let executed_ladder = graded_fixtures
            .iter()
            .filter(|g| g.id.starts_with("ladder-"))
            .map(|g| g.id.clone())
            .collect();
        let record = TranscribeRecord {
            schema: synapse_certify::RECORD_SCHEMA,
            row_id: row.into(),
            source_commit: source.into(),
            machine,
            executed_artifacts,
            model: speech::MODEL.into(),
            operation: "transcribe".into(),
            profile_id,
            fingerprint: pin.fingerprint,
            fixture_set_id: synapse_parity::evaluator::fixture_set_id(&manifest, speech::MODEL),
            precision_class: profile.precision_class,
            max_duration_ms: certified,
            max_encoder_frames: speech::frame_counts(certified * 16).2,
            graded_fixtures,
            expected_refusals: collected.refusals,
            executed_ladder,
            split_wer,
            self_check: collected
                .self_check
                .ok_or_else(|| refuse("self-check not observed"))?,
            measurements: collected
                .measurements
                .ok_or_else(|| refuse("speech measurements not observed"))?,
            observe_rerun: ObserveRerun {
                fixture_ids: collected.reruns,
                bit_identical: true,
            },
            status: "passed".into(),
        };
        validate_record(&record, &options.checkout, &options.assets, &catalog)?;
        Ok((record, catalog))
    }
    .await;
    let (record, catalog) = outcome.map_err(|e: synapse_certify::Error| refuse(format!(
        "transcribe certification incomplete for {row}; real hardware run required on {required}: {e}"
    )))?;
    write_record(&record, &options.checkout, &options.assets, &catalog)?;
    serde_json::to_value(record).map_err(err)
}

pub(crate) fn required_machine(row: &str) -> &'static str {
    match row {
        "metal-m5" | "ane-m5" => "M5",
        "cuda-windows-nvidia" | "vulkan-windows-nvidia" => "narli (Windows NVIDIA)",
        "cuda-linux-nvidia" | "vulkan-linux-nvidia" => "rented NVIDIA Linux",
        _ => "the declared AMD certification machine",
    }
}
