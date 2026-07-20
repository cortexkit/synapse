# MC paired sweep — 2026-07-20

> measured under representative ambient dev load (the daemon's production environment); queue-cleanliness proven via admission.status per cell, host-quiet not required

## Protocol and status

Both certified production lanes were exercised over the `subc` management surface with `embed.batch`, `input_type=document`, three sequential repetitions per cell, and one throwaway warmup per invocation. The client follows durable result pages and samples `admission.status` every 100 ms during each request. Each cell started only when loadavg-1m was below 8; loadavg-1m at every cell start is recorded below.

MEMORY fixtures measured 148–151 effective tokens/item; CHUNK fixtures submitted 3952–3954 tokens/item. The Metal lane processed CHUNK in full. ANE reported 512 effective tokens/item because its certified 512-token ceiling truncated the same submitted CHUNK inputs; submitted and effective numerators are both shown.

| Lane | Engine | Fingerprint | Module generation | Machine profile | OS build |
| --- | --- | --- | ---: | --- | --- |
| `gte-modernbert-base-f16` | owned-metal | `54a62ef80c4f28f6ba765854d81b9ab5e52d4864142cdd81662812465d3003b5` | 9 | `42a76cdd8dc2e5798629522c63dcfff1e5833ee1bf3c1f8bdb66dc2bbc04500d` | `25F84` |
| `gte-modernbert-base-ane-fp16` | ane-coreml-worker | `5a2374bcb587ae22cd7ca93404ee7e89e9889527d15f8671feb0a226625278d8` | 9 | `42a76cdd8dc2e5798629522c63dcfff1e5833ee1bf3c1f8bdb66dc2bbc04500d` | `25F84` |

## Full matrix

`items/s`, `tok/s`, and elapsed values are three-repetition medians; min–max columns expose the complete per-cell spread. `Acquire p50/p95` are the p95 of the rolling percentiles sampled during that cell. Power is populated for the required batch=64 CHUNK cell only.

| Lane | Class | Batch | Recommended | Loadavg 1m | Effective tok/item | Submitted tok/item | Elapsed ms min–max | Items/s median | Items/s min–max | Tok/s median | Tok/s min–max | Single p50/p95 ms | Waiters max | In-flight max | Acquire p50/p95 ms | GPU W | ANE W | Engine J/item |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Metal f16 | MEMORY | 1 | — | 7.18 | 150.0 | 150.0 | 34.04–34.71 | 29.22 | 28.81–29.38 | 4382.57 | 4321.86–4406.35 | 34.23 / 34.71 | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| Metal f16 | MEMORY | 2 | — | 7.18 | 149.0 | 149.0 | 35.05–35.91 | 56.25 | 55.70–57.06 | 8381.45 | 8299.53–8501.82 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| Metal f16 | MEMORY | 4 | — | 7.18 | 149.0 | 149.0 | 34.64–36.27 | 112.63 | 110.29–115.47 | 16782.09 | 16432.77–17204.96 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| Metal f16 | MEMORY | 8 | YES — 8 rows / 3072-token cap | 7.18 | 149.0 | 149.0 | 35.98–37.02 | 218.67 | 216.10–222.37 | 32581.92 | 32198.41–33133.08 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| Metal f16 | MEMORY | 16 | — | 7.18 | 148.0 | 148.0 | 72.38–73.54 | 219.08 | 217.58–221.05 | 32423.68 | 32201.22–32715.09 | — | 0 | 0 | 0.000292 / 0.000625 | — | — | — |
| Metal f16 | MEMORY | 32 | — | 7.18 | 149.0 | 149.0 | 143.13–145.36 | 220.26 | 220.15–223.57 | 32818.95 | 32802.22–33312.64 | — | 0 | 1 | 0.000292 / 0.000625 | — | — | — |
| Metal f16 | MEMORY | 64 | — | 7.18 | 149.0 | 149.0 | 303.23–351.62 | 210.24 | 182.01–211.06 | 31326.25 | 27120.12–31447.75 | — | 0 | 1 | 0.000292 / 0.000625 | — | — | — |
| Metal f16 | MEMORY | 128 | — | 7.18 | 151.0 | 151.0 | 602.48–676.96 | 212.02 | 189.08–212.45 | 32015.66 | 28551.18–32080.64 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| Metal f16 | MEMORY | 256 | — | 7.09 | 149.0 | 149.0 | 1203.98–1208.64 | 211.91 | 211.81–212.63 | 31574.33 | 31559.41–31681.59 | — | 0 | 1 | 0.000334 / 0.000625 | — | — | — |
| Metal f16 | CHUNK | 1 | — | 7.34 | 3953.0 | 3953.0 | 1587.99–2071.31 | 0.54 | 0.48–0.63 | 2148.52 | 1908.45–2489.31 | 1839.87 / 2071.31 | 0 | 1 | 0.000334 / 0.000708 | — | — | — |
| Metal f16 | CHUNK | 2 | — | 7.15 | 3952.0 | 3952.0 | 3152.66–3335.96 | 0.62 | 0.60–0.63 | 2449.29 | 2369.33–2507.09 | — | 0 | 1 | 0.000334 / 0.000708 | — | — | — |
| Metal f16 | CHUNK | 4 | — | 7.13 | 3953.0 | 3953.0 | 6385.07–6900.19 | 0.60 | 0.58–0.63 | 2371.05 | 2291.53–2476.40 | — | 0 | 1 | 0.000334 / 0.000708 | — | — | — |
| Metal f16 | CHUNK | 8 | YES — 8 rows / 3072-token cap | 6.30 | 3953.0 | 3953.0 | 14306.74–15262.24 | 0.53 | 0.52–0.56 | 2083.78 | 2072.04–2210.43 | — | 0 | 1 | 0.000334 / 0.000709 | — | — | — |
| Metal f16 | CHUNK | 16 | — | 7.56 | 3952.0 | 3952.0 | 28593.92–29449.36 | 0.55 | 0.54–0.56 | 2173.31 | 2147.14–2211.38 | — | 0 | 1 | 0.000375 / 0.000750 | — | — | — |
| Metal f16 | CHUNK | 32 | — | 7.37 | 3952.0 | 3952.0 | 57539.12–65442.28 | 0.51 | 0.49–0.56 | 2028.17 | 1932.45–2197.88 | — | 0 | 1 | 0.000417 / 0.001167 | — | — | — |
| Metal f16 | CHUNK | 64 | — | 7.96 | 3953.0 | 3953.0 | 121023.21–121406.26 | 0.53 | 0.53–0.53 | 2083.89 | 2083.85–2090.44 | — | 0 | 1 | 0.000666 / 0.001375 | 12.89 | 0.00 | 24.432 |
| Metal f16 | CHUNK | 128 | — | 6.17 | 3953.0 | 3953.0 | 209930.47–228692.45 | 0.56 | 0.56–0.61 | 2226.10 | 2212.51–2410.25 | — | 0 | 2 | 0.000625 / 0.001250 | — | — | — |
| Metal f16 | CHUNK | 256 | — | 7.37 | 3952.0 | 3952.0 | 467699.06–530274.16 | 0.52 | 0.48–0.55 | 2060.60 | 1907.90–2163.17 | — | 0 | 2 | 0.000708 / 0.001500 | — | — | — |
| ANE fp16 | MEMORY | 1 | — | 7.09 | 149.0 | 149.0 | 10.20–10.85 | 97.61 | 92.14–98.00 | 14544.33 | 13728.92–14602.41 | 10.24 / 10.85 | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| ANE fp16 | MEMORY | 2 | — | 7.09 | 149.0 | 149.0 | 19.27–19.52 | 103.72 | 102.46–103.80 | 15454.63 | 15266.33–15465.46 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| ANE fp16 | MEMORY | 4 | — | 7.09 | 149.0 | 149.0 | 36.73–37.41 | 108.05 | 106.93–108.89 | 16099.91 | 15932.17–16224.44 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| ANE fp16 | MEMORY | 8 | — | 7.09 | 149.0 | 149.0 | 71.82–72.46 | 110.93 | 110.40–111.39 | 16528.67 | 16449.59–16597.27 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| ANE fp16 | MEMORY | 16 | — | 7.09 | 148.0 | 148.0 | 143.08–144.23 | 111.77 | 110.94–111.82 | 16542.08 | 16418.55–16549.90 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| ANE fp16 | MEMORY | 32 | — | 6.68 | 149.0 | 149.0 | 286.40–286.85 | 111.71 | 111.56–111.73 | 16644.45 | 16621.84–16648.10 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| ANE fp16 | MEMORY | 64 | — | 6.68 | 149.0 | 149.0 | 601.74–602.75 | 106.22 | 106.18–106.36 | 15827.39 | 15820.90–15847.25 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| ANE fp16 | MEMORY | 128 | — | 6.68 | 149.0 | 149.0 | 1170.22–1175.22 | 109.35 | 108.92–109.38 | 16292.68 | 16228.42–16297.78 | — | 0 | 1 | 0.000375 / 0.000625 | — | — | — |
| ANE fp16 | MEMORY | 256 | — | 6.79 | 150.0 | 150.0 | 2312.74–2321.36 | 110.50 | 110.28–110.69 | 16574.98 | 16542.00–16603.71 | — | 0 | 1 | 0.000333 / 0.000625 | — | — | — |
| ANE fp16 | CHUNK | 1 | — | 7.84 | 512.0 | 3952.0 | 30.17–36.69 | 28.52 | 27.25–33.15 | 14603.31 | 13954.33–16972.54 | 35.06 / 36.69 | 0 | 0 | 0.000417 / 0.001084 | — | — | — |
| ANE fp16 | CHUNK | 2 | — | 7.84 | 512.0 | 3952.0 | 57.17–58.69 | 34.68 | 34.08–34.98 | 17757.74 | 17446.71–17912.22 | — | 0 | 0 | 0.000417 / 0.001083 | — | — | — |
| ANE fp16 | CHUNK | 4 | — | 7.84 | 512.0 | 3953.0 | 112.54–113.82 | 35.33 | 35.14–35.54 | 18086.97 | 17992.54–18198.20 | — | 0 | 1 | 0.000417 / 0.001042 | — | — | — |
| ANE fp16 | CHUNK | 8 | — | 7.84 | 512.0 | 3952.0 | 224.91–225.24 | 35.52 | 35.52–35.57 | 18186.42 | 18184.86–18211.71 | — | 0 | 1 | 0.000416 / 0.001042 | — | — | — |
| ANE fp16 | CHUNK | 16 | — | 7.84 | 512.0 | 3953.0 | 447.07–448.29 | 35.72 | 35.69–35.79 | 18290.44 | 18273.89–18323.87 | — | 0 | 1 | 0.000416 / 0.001041 | — | — | — |
| ANE fp16 | CHUNK | 32 | — | 7.84 | 512.0 | 3953.0 | 906.72–926.29 | 34.73 | 34.55–35.29 | 17783.37 | 17687.83–18069.45 | — | 0 | 1 | 0.000375 / 0.000750 | — | — | — |
| ANE fp16 | CHUNK | 64 | — | 7.53 | 512.0 | 3952.0 | 1819.12–1848.20 | 35.10 | 34.63–35.18 | 17973.10 | 17729.64–18013.10 | — | 0 | 1 | 0.000416 / 0.000667 | 0.63 | 4.54 | 0.130 |
| ANE fp16 | CHUNK | 128 | — | 7.41 | 512.0 | 3954.0 | 3602.06–3624.34 | 35.46 | 35.32–35.54 | 18156.31 | 18082.19–18194.03 | — | 0 | 1 | 0.000416 / 0.000667 | — | — | — |
| ANE fp16 | CHUNK | 256 | — | 6.80 | 512.0 | 3952.0 | 7181.84–7222.00 | 35.60 | 35.45–35.65 | 18224.96 | 18148.98–18250.48 | — | 0 | 1 | 0.000375 / 0.000666 | — | — | — |

## Queue-cleanliness evidence

- All 36 cells recorded `execution_waiters_max=0`; no cell was averaged through daemon-side waiter contention.
- Cell loadavg-1m range: 6.17–7.96; no timed cell began at or above the 8.0 gate.
- `inline_in_flight_executions_max` was 0–2. The value 2 appeared on long Metal durable jobs without waiters; it is internal execution overlap, not evidence of a foreign queued request.
- Acquire-wait samples were present for every cell; the table reports rolling p50/p95 percentiles and their per-cell p95.

## Power and energy

- Metal batch=64 CHUNK: mean macmon GPU 12.89 W, ANE 0 W, engine energy 24.432 J/item over 617 in-window samples.
- ANE batch=64 CHUNK: mean macmon GPU 0.63 W, ANE 4.54 W, engine energy 0.130 J/item over 5 in-window samples.
- CPU/system power was sampled but is not used for the engine J/item column; macmon is sudo-free.

## Recommended-batch contract note

The live `models.list` response omitted `recommended_batch`, confirming the wire-contract gap: `ModelCatalogEntry` currently serializes only model id, state, and fingerprints. Per the documented era values, Metal batch 8 is marked as recommended for the 8-row / 3072-token coalescing cap. No ANE marker is applied because the ANE-WAVE documents available in this checkout specify the bucket ladder and placement policy but do not state a preferred request batch; the row is intentionally omitted rather than inferred.

## Honest notes

- Two preliminary Metal attempts were correctly aborted when the corrected load gate observed 9.82 and 10.17 loadavg-1m. Their partial outputs were discarded.
- The final cellwise runner waited for loadavg-1m below 8 before each cell and reran cells with `execution_waiters_max>0`; all accepted cells were waiter-free.
- The Metal CHUNK lane is much slower than ANE here because it processed ~3952 effective tokens/item versus ANE's 512-token truncation. Compare submitted/effective token columns before interpreting throughput.
- No daemon restart, redeploy, model load, or probe was performed. Both lanes remained certified and non-stale throughout.
