# CodeGraph `explore` dump vs. learned gather

Date: 2026-07-16

## Verdict

**CodeGraph's deterministic one-shot dump loses on both citation overlap and consumer package economy.** It returned a mechanically valid package for all 40 questions, but its mean file F1 was **0.0616** versus the 4B trained gatherer's published natural-only **0.637** and the gold identity baseline's **1.000**. Its mean hydrated package was **4,989.9 o200k tokens**, 1.84× the gold mean, while producing only **0.0124 F1 points per 1k package tokens**. Gold reaches **0.3687** on the same metric: 29.9× the quality per token.

The deterministic query is quick and needs no GPU or LLM—mean `codegraph_explore` call time was **565.4 ms**—but it is not a gather replacement for this corpus. It selected graph-near code that was often unrelated to the question's gold citations and hydrated broad source ranges into packages the consumer must read.

## Protocol

- **CodeGraph:** v1.4.1 at `246aee837341183912c82b3e727410e9fe1a1567`.
- **Corpus:** the 40 fixed jobs from `data/eval-jobs.jsonl`, against the 40 Opus gold rows in `data/eval-gold-rows.jsonl`.
- **Isolation:** each of the five read-only corpus repositories was copied to `/tmp/codegraph-eval/<owner>__<repo>`. CodeGraph wrote its `.codegraph/` index only in those copies. Package paths remain repository-relative, so validation and scoring resolve them against the original pinned `~/Work/OSS/gather-corpus-eval` clones.
- **Invocation:** one `codegraph_explore` MCP call per question, with the request passed verbatim as `query`; no LLM, retry synthesis, secondary retrieval, or post-filtering was used. CodeGraph's default adaptive file limit was left unchanged.
- **Pin handling:** the shared CodeGraph checkout had advanced after the experiment pin was recorded. The requested commit remained fetchable, so the run used an isolated scratch checkout at the recorded SHA. The source checkout did not carry generated `dist/`, so that scratch copy alone was built with its supplied `node_modules`; the shared CodeGraph checkout and its generated output were not changed.
- **Validation and score:** all 40 rows passed the normal `validate` lane, then the normal `score` lane revalidated them and scored citations against gold. The output files are `data/students/codegraph-explore-rows.jsonl`, `data/students/codegraph-explore-scores.json`, and `data/students/codegraph-explore-package-metrics.json`. Raw MCP replies are retained in `data/students/codegraph-explore-raw.jsonl`.

`natural` has a different meaning for this row than it has for model trajectories: every CodeGraph call that adapted to a valid package is marked natural because it has no generation budget or forced-finalization state. It must not be read as a learned gather completion-rate result.

## Adapter mapping

| CodeGraph MCP output | Gather package field | Mapping |
| --- | --- | --- |
| Caller query | `interpretation` | The question is preserved verbatim. CodeGraph does not emit an independent interpretation sentence. |
| `**\`path\`** — symbol/relationship description` file heading, plus `path:line` relation and blast-radius references | `scope`, `snippets[].path`, `snippets[].why` | Every named repository file joins `scope`; a source heading's reported symbol/relationship text becomes that snippet's `why`. |
| Numbered source lines inside that file's fenced block | `snippets[].startLine`, `snippets[].endLine` | Every contiguous line-number run becomes one snippet. A gap creates another snippet; the adapter never bridges omitted source or clamps a range. |
| No omission analysis in the dump | `omissions` | Always `[]`. This is a structural difference from learned gather. |
| MCP error or no source block with a faithful path/range | row validity | `final_json: null` and an honest failed/invalid row; it is not padded with guessed citations. |

The adapter records `snippet_bytes`, hydrated `o200k_base` tokens, snippet count, and wall time in each row's `codegraph_explore` metadata. It performs inline pinned-clone validation before staging rows. The score lane then repeats validation independently.

## Full adapter example: raw dump and mapped package

Question: “Trace how a value produced by the Rust core is serialized and eventually consumed by `clients/store/src/derivation.ts` — what intermediate format or crate boundary connects them?”

The raw reply below contains the complete MCP text for job `c4022f7663e63ff5473919251ddc7beb1eefc5ac667e2a3c42531728b1f8afb9` (with trailing tabs on otherwise blank displayed source lines removed only for Markdown cleanliness); it selected one 367-line block in `crates/subc-core/src/bin/ck.rs`. The adjacent mapped package preserves that exact range and its CodeGraph-provided role string. It also illustrates the failure mode: the dump follows a broad `ck.rs` cluster rather than the requested Rust-to-TypeScript boundary.

### Raw `codegraph_explore` response

<details>
<summary>Complete raw Markdown returned by CodeGraph; exact bytes are retained in the raw JSONL artifact</summary>

~~~~text
**Dynamic-dispatch links among your symbols**
(synthesized — the indirect hops grep/Read would reconstruct; the `@file:line` is the wiring site)

- handle → handle   [dynamic: interface → impl @crates/subc-client-rs/examples/echo-module.rs:41]

> Full source for these symbols is below — the call flow among them, followed by their bodies.
**Exploration: Trace how a value produced by the Rust core is serialized and eventually consumed by clients/store/src/derivation.ts — what intermediate format or crate boundary connects them?**

Found 24 symbols across 1 file.

**Blast radius — what depends on these (update/verify before editing)**

- `connect` (clients/subc-client/src/provider.ts:560) — 3 callers; tests: `clients/subc-client/tests/live-provider.test.ts`, `clients/subc-client/tests/live-streaming.test.ts`, `clients/subc-client/tests/provider.test.ts`
- `connect` (clients/subc-client/src/socket.ts:93) — 3 callers in `clients/subc-client/src/client.ts`, `clients/subc-client/src/provider.ts`; tests: `clients/subc-client/tests/socket.test.ts`
- `connect` (clients/subc-client/src/client.ts:335) — 5 callers; tests: `clients/subc-client/tests/close-route.test.ts`, `clients/subc-client/tests/live-handshake.test.ts`, `clients/subc-client/tests/live-provider.test.ts`, `clients/subc-client/tests/live-streaming.test.ts` +1
- `connect` (clients/subc-client-swift/Sources/SubcClient/Client.swift:107) — 5 callers in `clients/subc-client-swift/Sources/SubcChat/ChatViewModel.swift`, `clients/subc-client-swift/Sources/SubcChat/ObserveViewModel.swift`, `clients/subc-client-swift/Sources/SubcChat/RoomsViewModel.swift`, `clients/subc-client-swift/Sources/SubcClient/Transport.swift`; ⚠️ no covering tests found
- `connect` (crates/subc-client-rs/src/consumer.rs:321) — 5 callers; tests: `crates/subc-client-rs/tests/real_daemon.rs`

**Source Code**

> The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.

**`crates/subc-core/src/bin/ck.rs`** — calls(calls), CkError(references), Result(references), references(references), instantiates(instantiates), +27 more

```rust
224	    epoch: u32,
225	}
226
227	struct CkClient {
228	    path: PathBuf,
229	    info: ConnectionInfo,
230	    stream: TcpStream,
231	    next_corr: u64,
232	}
233
234	impl CkClient {
235	    async fn connect(resolved: ResolvedConnection) -> Result<Self, CkError> {
236	        let endpoint = resolved
237	            .info
238	            .endpoints
239	            .first()
240	            .ok_or_else(|| CkError::Connection {
241	                path: resolved.path.clone(),
242	                source: "connection file has no endpoints".to_string(),
243	            })?;
244	        let ip: IpAddr = endpoint.host.parse().map_err(|_| CkError::Connection {
245	            path: resolved.path.clone(),
246	            source: format!("endpoint host is not an IP: {}", endpoint.host),
247	        })?;
248	        let addr = SocketAddr::new(ip, endpoint.port);
249	        let mut stream = match time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addr)).await {
250	            Ok(Ok(stream)) => stream,
251	            Ok(Err(source)) => {
252	                return Err(CkError::Connection {
253	                    path: resolved.path,
254	                    source: format!("connect {addr}: {source}"),
255	                })
256	            }
257	            Err(_) => {
258	                return Err(CkError::Connection {
259	                    path: resolved.path,
260	                    source: format!("connect {addr}: timed out after {CONNECT_TIMEOUT:?}"),
261	                })
262	            }
263	        };
264	        authenticate_client(&mut stream, &resolved.info, AUTH_DEADLINE)
265	            .await
266	            .map_err(|source| CkError::Connection {
267	                path: resolved.path.clone(),
268	                source: format!("authenticate: {source}"),
269	            })?;
270
271	        Ok(Self {
272	            path: resolved.path,
273	            info: resolved.info,
274	            stream,
275	            next_corr: 1,
276	        })
277	    }
278
279	    async fn rpc_value(&mut self, request: ClientControlRequest) -> Result<Value, CkError> {
280	        let frame = self.rpc_frame(request).await?;
281	        match frame.header.ty {
282	            FrameType::Response => Ok(serde_json::from_slice(&frame.body)?),
283	            FrameType::Error => Err(CkError::Rejected(decode_error_body(&frame.body))),
284	            ty => Err(CkError::Message(format!(
285	                "unexpected control response frame {ty:?}"
286	            ))),
287	        }
288	    }
289
290	    async fn rpc_frame(&mut self, request: ClientControlRequest) -> Result<Frame, CkError> {
291	        let corr = self.next_corr;
292	        self.next_corr = self.next_corr.saturating_add(1);
293	        let body = serde_json::to_vec(&request)?;
294	        let frame = Frame::build(
295	            FrameType::Request,
296	            Flags::new(false, Priority::Interactive, false),
297	            0,
298	            0,
299	            corr,
300	            body,
301	        )
302	        .map_err(|source| CkError::Message(source.to_string()))?;
303	        write_frame(&mut self.stream, &frame)
304	            .await
305	            .map_err(|source| CkError::Message(source.to_string()))?;
306
307	        loop {
308	            let reply = self.next_frame().await?;
309	            if reply.header.channel == 0
310	                && reply.header.corr == corr
311	                && matches!(reply.header.ty, FrameType::Response | FrameType::Error)
312	            {
313	                return Ok(reply);
314	            }
315	        }
316	    }
317
318	    async fn next_frame(&mut self) -> Result<Frame, CkError> {
319	        match time::timeout(RESPONSE_TIMEOUT, read_frame(&mut self.stream)).await {
320	            Ok(Ok(Some(frame))) => Ok(frame),
321	            Ok(Ok(None)) => Err(CkError::Message("subc closed the connection".into())),
322	            Ok(Err(source)) => Err(CkError::Message(format!("read frame: {source}"))),
323	            Err(_) => Err(CkError::Message(format!(
324	                "timed out after {RESPONSE_TIMEOUT:?} waiting for a frame"
325	            ))),
326	        }
327	    }
328
329	    async fn catalog_list(&mut self) -> Result<Vec<CatalogEntry>, CkError> {
330	        let value = self
331	            .rpc_value(ClientControlRequest::CatalogList { module_id: None })
332	            .await?;
333	        match serde_json::from_value::<ClientControlResponse>(value)? {
334	            ClientControlResponse::CatalogList { modules, .. } => Ok(modules),
335	            other => Err(CkError::Message(format!(
336	                "unexpected catalog.list response: {other:?}"
337	            ))),
338	        }
339	    }
340
341	    async fn route_open_management(
342	        &mut self,
343	        module_id: &str,
344	        project_root: PathBuf,
345	    ) -> Result<RouteHandle, CkError> {
346	        let request = ClientControlRequest::RouteOpen {
347	            target: RouteTarget::ManagementSurface {
348	                module_id: module_id.to_string(),
349	            },
350	            identity: BindIdentity {
351	                project_root,
352	                harness: CK_HARNESS.to_string(),
353	                session: "quota".to_string(),
354	            },
355	            consumer_identity: None,
356	            consumer_capabilities: None,
357	        };
358	        let value = self.rpc_value(request).await?;
359	        match serde_json::from_value::<ClientControlResponse>(value)? {
360	            ClientControlResponse::RouteOpen {
361	                route_channel,
362	                route_epoch,
363	            } => Ok(RouteHandle {
364	                channel: route_channel,
365	                epoch: route_epoch,
366	            }),
367	            other => Err(CkError::Message(format!(
368	                "unexpected route.open response: {other:?}"
369	            ))),
370	        }
371	    }
372
373	    async fn route_request_value(
374	        &mut self,
375	        route: RouteHandle,
376	        body: Value,
377	    ) -> Result<Value, CkError> {
378	        let corr = self.next_corr;
379	        self.next_corr = self.next_corr.saturating_add(1);
380	        let body = serde_json::to_vec(&body)?;
381	        let frame = Frame::build(
382	            FrameType::Request,
383	            Flags::new(false, Priority::Interactive, false),
384	            route.channel,
385	            route.epoch,
386	            corr,
387	            body,
388	        )
389	        .map_err(|source| CkError::Message(source.to_string()))?;
390	        write_frame(&mut self.stream, &frame)
391	            .await
392	            .map_err(|source| CkError::Message(source.to_string()))?;
393
394	        loop {
395	            let reply = self.next_frame().await?;
396	            if reply.header.channel != route.channel
397	                || reply.header.epoch != route.epoch
398	                || reply.header.corr != corr
399	            {
400	                continue;
401	            }
402	            return match reply.header.ty {
403	                FrameType::Response => Ok(serde_json::from_slice(&reply.body)?),
404	                FrameType::Error => Err(CkError::Rejected(decode_error_body(&reply.body))),
405	                ty => Err(CkError::Message(format!(
406	                    "unexpected route response frame {ty:?}"
407	                ))),
408	            };
409	        }
410	    }
411
412	    async fn route_goodbye(&mut self, route: RouteHandle) {
413	        let frame = match Frame::build(
414	            FrameType::Goodbye,
415	            Flags::new(false, Priority::Passive, false),
416	            route.channel,
417	            route.epoch,
418	            0,
419	            Vec::new(),
420	        ) {
421	            Ok(frame) => frame,
422	            Err(_) => return,
423	        };
424	        let _ = write_frame(&mut self.stream, &frame).await;
425	    }
426	}
427
428	async fn module_list(client: &mut CkClient, json_output: bool) -> Result<(), CkError> {
429	    let value = supervisor_list(client).await?;
430	    if json_output {
431	        print_json(&value)?;
432	    } else {
433	        print_module_table(modules_array(&value));
434	    }
435	    Ok(())
436	}
437
438	async fn module_status(
439	    client: &mut CkClient,
440	    module_id: &str,
441	    json_output: bool,
442	) -> Result<(), CkError> {
443	    let list = supervisor_list(client).await?;
444	    let module = find_module(&list, module_id)
445	        .cloned()
446	        .ok_or_else(|| CkError::Rejected(format!("module_id '{module_id}' is not supervised")))?;
447	    let health = supervisor_health(client).await?;
448	    let health_entry = find_module(&health, module_id).cloned();
449
450	    if json_output {
451	        print_json(&json!({ "module": module, "health": health_entry }))?;
452	    } else {
453	        print_status_table(&module, health_entry.as_ref());
454	    }
455	    Ok(())
456	}
457
458	async fn module_restart(
459	    client: &mut CkClient,
460	    module_id: &str,
461	    json_output: bool,
462	) -> Result<(), CkError> {
463	    let ack = client
464	        .rpc_value(ClientControlRequest::SupervisorRestart {
465	            module_id: module_id.to_string(),
466	        })
467	        .await?;
468	    print_ack_with_state(client, module_id, ack, "restart", json_output).await
469	}
470
471	async fn module_rescan(client: &mut CkClient, json_output: bool) -> Result<(), CkError> {
472	    let result = client
473	        .rpc_value(ClientControlRequest::SupervisorRescan {})
474	        .await?;
475	    if json_output {
476	        print_json(&result)?;
477	    } else {
478	        print_rescan_table(&result);
479	    }
480	    Ok(())
481	}
482
483	async fn module_set_enabled(
484	    client: &mut CkClient,
485	    module_id: &str,
486	    enabled: bool,
487	    json_output: bool,
488	) -> Result<(), CkError> {
489	    let ack = client
490	        .rpc_value(ClientControlRequest::SupervisorSetEnabled {
491	            module_id: module_id.to_string(),
492	            enabled,
493	        })
494	        .await?;
495	    let verb = if enabled { "start" } else { "stop" };
496	    print_ack_with_state(client, module_id, ack, verb, json_output).await
497	}
498
499	async fn print_ack_with_state(
500	    client: &mut CkClient,
501	    module_id: &str,
502	    ack: Value,
503	    verb: &str,
504	    json_output: bool,
505	) -> Result<(), CkError> {
506	    let list = supervisor_list(client).await?;
507	    let module = find_module(&list, module_id).cloned();
508	    let state = module
509	        .as_ref()
510	        .and_then(|value| value.get("state"))
511	        .and_then(Value::as_str)
512	        .unwrap_or("-");
513	    let applied = ack
514	        .get("applied")
515	        .and_then(Value::as_bool)
516	        .ok_or_else(|| CkError::Message(format!("unexpected {verb} ack: {ack}")))?;
517
518	    if json_output {
519	        let mut output = ack;
520	        if let Some(object) = output.as_object_mut() {
521	            object.insert("state".to_string(), Value::String(state.to_string()));
522	            object.insert(
523	                "module".to_string(),
524	                module.unwrap_or_else(|| Value::Object(Default::default())),
525	            );
526	        }
527	        print_json(&output)?;
528	    } else {
529	        print_table(
530	            &["module", "applied", "state"],
531	            vec![vec![
532	                module_id.to_string(),
533	                applied.to_string(),
534	                state.to_string(),
535	            ]],
536	        );
537	    }
538	    Ok(())
539	}
540
541	async fn health(client: &mut CkClient, json_output: bool) -> Result<(), CkError> {
542	    let value = supervisor_health(client).await?;
543	    if json_output {
544	        print_json(&value)?;
545	    } else {
546	        print_health_table(modules_array(&value));
547	    }
548	    Ok(())
549	}
550
551	async fn daemon(client: &mut CkClient, json_output: bool) -> Result<(), CkError> {
552	    let connected_clients = client
553	        .rpc_value(ClientControlRequest::ServerDescribe {})
554	        .await?;
555	    if json_output {
556	        print_json(&connected_clients)?;
557	    } else {
558	        let uptime = connection_file_age(&client.path)
559	            .map(format_duration)
560	            .unwrap_or_else(|| "-".to_string());
561	        let protocol = display_field(&connected_clients, "protocol_ver");
562	        let clients = display_field(&connected_clients, "connected_clients");
563	        print_table(
564	            &[
565	                "daemon_ver",
566	                "protocol",
567	                "pid",
568	                "connected_clients",
569	                "uptime",
570	            ],
571	            vec![vec![
572	                client.info.daemon_ver.clone(),
573	                protocol,
574	                client.info.pid.to_string(),
575	                clients,
576	                uptime,
577	            ]],
578	        );
579	    }
580	    Ok(())
581	}
582
583	async fn supervisor_list(client: &mut CkClient) -> Result<Value, CkError> {
584	    client
585	        .rpc_value(ClientControlRequest::SupervisorList {})
586	        .await
587	}
588
589	async fn supervisor_health(client: &mut CkClient) -> Result<Value, CkError> {
590	    client
```


... (output truncated to budget; the source above is complete and verbatim — treat it as already Read. For any area not covered, run another codegraph_explore with the specific names — do NOT Read these files.)
~~~~

</details>

### Mapped gather package

```json
{
  "interpretation": "Trace how a value produced by the Rust core is serialized and eventually consumed by clients/store/src/derivation.ts — what intermediate format or crate boundary connects them?",
  "scope": [
    "crates/subc-core/src/bin/ck.rs",
    "crates/subc-client-rs/examples/echo-module.rs",
    "clients/subc-client/src/provider.ts",
    "clients/subc-client/src/socket.ts",
    "clients/subc-client/src/client.ts",
    "clients/subc-client-swift/Sources/SubcClient/Client.swift",
    "crates/subc-client-rs/src/consumer.rs"
  ],
  "snippets": [
    {
      "path": "crates/subc-core/src/bin/ck.rs",
      "startLine": 224,
      "endLine": 590,
      "why": "codegraph_explore calls(calls), CkError(references), Result(references), references(references), instantiates(instantiates), +27 more"
    }
  ],
  "omissions": []
}
```

## F1 result

F1 and line Jaccard use the existing scorer's cited-file and inclusive-line logic. The CodeGraph row includes every valid package because all 40 are natural by adapter construction.

| system | F1 basis | file F1 | line Jaccard | valid packages | natural packages |
| --- | --- | ---: | ---: | ---: | ---: |
| `codegraph-explore` | scored 40/40 | **0.0616** | 0.1057 | 40/40 | 40/40 |
| `qwen35-4b-lora-v1` | published natural-only ladder result | **0.637** | 0.639 | 35/40 | 37/40 |
| Opus gold | identity reference | 1.0000 | 1.0000 | 40/40 | 40/40 |

The trained 4B's raw `*-rows.jsonl` was not retained in the parent `data/students/` directory made available to this run, so its F1 is cited from `LADDER.md` and its validity count is the published 87.5%, not recomputed here. No DeepSeek row file was present either. This report does not invent hydrated size or quality-per-token values for unavailable rows.

## Size is a co-headline metric

The token count is a real tokenizer measurement, not bytes/4: Python `tiktoken` `o200k_base` tokenizes exactly `JSON.stringify(hydrateJudgePackage(package), null, 2)`. That hydrated serialization contains the interpretation, scope, each snippet's path/range/why, every exact source byte, and omissions—the package that the utility judge receives. P95 uses nearest-rank order statistics.

| system | hydrated packages measured | snippets/package mean / median / p95 | snippet bytes mean / median / p95 | hydrated o200k tokens mean / median / p95 | wall time/package mean / median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `codegraph-explore` | 40 | 7.45 / 7 / 17 | 15,543 / 14,997 / 19,964 | **4,990 / 5,014 / 6,543** | **565.4 / 485 / 1,135 ms** |
| `qwen35-4b-lora-v1` | unavailable | unavailable | unavailable | unavailable | published 72.5 s/trajectory |
| Opus gold | 40 | 6.73 / 6 / 13 | 7,339 / 4,087 / 24,376 | **2,712 / 1,919 / 8,043** | not recorded in gold rows |

The CodeGraph timing is deterministic local explore-call time only; it excludes the one-time indexing of each scratch copy. The raw response itself is also not charged to the consumer package: the consumer sees the hydrated package, as the judge does.

| system | file F1 | mean package tokens | F1 points / 1k mean package tokens |
| --- | ---: | ---: | ---: |
| `codegraph-explore` | 0.0616 | 4,989.9 | **0.0124** |
| `qwen35-4b-lora-v1` | 0.6370 | unavailable | unavailable |
| Opus gold | 1.0000 | 2,712.5 | **0.3687** |

CodeGraph's mean package is 1.84× gold's, but its raw F1 is only 6.16% of gold's. After size normalization it therefore **loses**, not ties: gold yields 29.9× more file-F1 per hydrated token. The trained 4B already has 10.3× CodeGraph's raw F1 before any size advantage is considered; a numerical 4B quality-per-token claim awaits the missing row artifact rather than an assumed “2–6 snippets” value.

## Limitations

1. **No omissions reasoning.** The dump says what it retrieved but cannot state which relevant lead it knowingly skipped, so every mapped package has an empty omissions list.
2. **No interpretation or query planning.** The verbatim question is metadata, not a learned explanation of the evidence need. There is no iterative correction when the first graph neighborhood is wrong.
3. **Retrieval heuristic dependence.** `codegraph_explore` selects an indexed graph neighborhood and may return broad source clusters or symbols connected by generic names. The example above is valid source evidence yet fails the requested cross-language trace.
4. **Rendering limits.** The MCP text renderer caps output and can show focused ranges or source gaps. The adapter preserves only rendered numbered ranges; it cannot recover omitted graph evidence.
5. **Comparison-data availability.** Gold and CodeGraph packages were hydrated from retained row files. The 4B and optional DeepSeek row files were absent, so their package-size columns are explicitly unavailable rather than estimated.

The result is a clear placement: deterministic graph retrieval is fast and mechanically reliable, but on this benchmark it lands below both the learned gatherer and gold on citation quality and below gold on quality per consumer token.
