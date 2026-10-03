# Does a gap between direct-API ANE dispatches remove slow bursts?

Measured 2026-10-03 on an Apple M5 Max, macOS 27.0.1 (26A434). The model was the
full `Alibaba-NLP/gte-modernbert-base` encoder through the private
`_ANEInMemoryModel` path (`bench/spikes/ane-direct-probe`, binary
`modernbert_full`) at sequence 512. One prediction is 13 `run_cached` dispatches:
the embedding norm, 11 executables of two fused layers each (the 512 shape
compiles with two layers per executable whatever was asked for), and the final
norm.

## Verdict: clean negative

A 3 ms sleep between predictions did not reduce slow calls. Neither arm showed
the 300-700 ms `aned` scheduling stall that anemll-forge reports for Core AI on
M6 when calls are issued less than about 2 ms apart. Across 10,000 predictions
(130,000 dispatches) the slowest prediction took 78 ms and the slowest single
dispatch 12 ms. Back-to-back dispatch on this path does not trigger that stall.

The slow stretches we did see were not isolated stalls. They were plateaus that
lasted tens to hundreds of consecutive calls at 30-40 ms instead of 25 ms, they
appeared in both arms (block 16 with no gap; blocks 13, 15 and 19 with the 3 ms
gap), and they began as the machine's load rose from 10 to about 29 because of
other work. That pattern points to contention for shared resources rather than
dispatch spacing. Treat this as an explanation the data supports, not a proven
cause.

The gap cost throughput. With the gap counted, the 3 ms arm completed 16% fewer
predictions per second (32.3/s against 38.5/s). Without it, 3.6% fewer
(36.7/s against 38.5/s). The per-row medians were about 0.4 ms slower with the
gap, for every row.

Because the 3 ms arm did not show fewer stall-signature calls, the planned
follow-up at 1 ms was not run.

## Per arm

20 blocks of 500 predictions alternate 0 ms, 3 ms, 0 ms, and so on. Each block
cycles through the 8 rows of `rows.jsonl` in file order. Times are wall clock
per prediction: CPU embedding gather, IOSurface conversion and all 13 dispatches
are included, and the gap sleep is not. "Stall signature" means a call slower
than 3x the arm's median.

| arm | blocks | calls | median | p95 | p99 | max | stall signature (>3x arm median) | >1.5x arm median | dispatches >3x their position's median | max dispatch | calls/s, gap excluded | calls/s, gap included |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 ms | 10 | 5000 | 25.04 ms | 32.29 | 33.90 | 44.66 | 0 | 12 | 29 / 65000 | 11.29 ms | 38.49 | 38.48 |
| 3 ms | 10 | 5000 | 25.46 ms | 35.43 | 43.39 | 78.23 | 1 | 161 | 50 / 65000 | 12.00 ms | 36.69 | 32.26 |

The requested 3 ms sleep measured 3.76 ms at the median (11.7 ms at worst),
which is ordinary `thread::sleep` overshoot. Every gap was therefore above the
reported 2 ms threshold.

## Per block, with load

| block | arm | 1m load start→end | 5m load at start | median | p95 | p99 | max | >3x block median |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 ms | 11.61→15.34 | 10.67 | 24.78 | 26.33 | 27.10 | 28.75 | 0 |
| 1 | 3 ms | 15.34→18.03 | 11.53 | 26.54 | 31.00 | 33.02 | 36.10 | 0 |
| 2 | 0 ms | 18.03→18.60 | 12.40 | 25.58 | 28.75 | 32.46 | 32.96 | 0 |
| 3 | 3 ms | 18.60→18.11 | 12.72 | 24.80 | 25.93 | 26.48 | 29.74 | 0 |
| 4 | 0 ms | 18.11→22.08 | 12.89 | 24.68 | 26.86 | 27.98 | 29.08 | 0 |
| 5 | 3 ms | 22.08→20.55 | 13.90 | 24.79 | 26.84 | 27.63 | 31.10 | 0 |
| 6 | 0 ms | 20.55→30.97 | 13.97 | 24.77 | 25.89 | 26.99 | 31.27 | 0 |
| 7 | 3 ms | 30.97→27.88 | 16.57 | 24.61 | 25.24 | 25.81 | 27.12 | 0 |
| 8 | 0 ms | 27.88→27.43 | 16.59 | 24.67 | 25.84 | 27.23 | 30.39 | 0 |
| 9 | 3 ms | 27.43→26.49 | 16.87 | 24.94 | 26.25 | 26.79 | 27.27 | 0 |
| 10 | 0 ms | 26.49→23.95 | 17.20 | 24.84 | 26.34 | 26.79 | 27.16 | 0 |
| 11 | 3 ms | 23.95→23.95 | 17.10 | 24.76 | 26.48 | 27.18 | 27.44 | 0 |
| 12 | 0 ms | 23.95→21.60 | 17.32 | 24.80 | 27.07 | 29.00 | 32.57 | 0 |
| 13 | 3 ms | 21.60→22.52 | 17.12 | 29.32 | 43.89 | 46.16 | 54.50 | 0 |
| 14 | 0 ms | 22.52→21.95 | 17.51 | 25.48 | 27.98 | 28.98 | 30.46 | 0 |
| 15 | 3 ms | 21.95→26.02 | 17.63 | 30.31 | 35.18 | 36.13 | 43.21 | 0 |
| 16 | 0 ms | 26.02→27.50 | 18.74 | 32.05 | 35.43 | 39.41 | 44.66 | 0 |
| 17 | 3 ms | 27.50→29.48 | 19.55 | 28.60 | 35.36 | 37.94 | 42.36 | 0 |
| 18 | 0 ms | 29.48→29.52 | 20.43 | 27.17 | 31.19 | 33.02 | 33.71 | 0 |
| 19 | 3 ms | 29.52→28.13 | 20.89 | 27.11 | 40.27 | 52.47 | 78.23 | 0 |

## Time series

`series.svg` plots every block's 500 per-call times in order on a shared log
axis, with a red line at 3x the arm's median. No point in any block crosses it
except one in block 19. Below, each block is split into ten windows of 50 calls,
shown as window median/maximum in ms, for three blocks of each arm:

| block | arm | calls 0-49 … 450-499 |
|---:|---|---|
| 0 | 0 ms | 25.4/27 25.5/27 25.6/27 24.6/29 24.6/26 24.5/26 24.4/25 24.6/27 25.1/27 24.7/26 |
| 12 | 0 ms | 24.5/27 24.7/27 24.6/26 24.5/26 24.7/25 24.9/29 25.0/26 24.8/28 25.0/26 26.9/33 |
| 16 | 0 ms | 26.8/32 24.8/26 31.7/41 32.9/38 32.6/45 32.7/44 30.3/32 28.6/34 32.8/36 33.0/36 |
| 1 | 3 ms | 25.2/28 25.0/26 25.1/27 29.3/36 29.1/33 27.3/28 27.0/28 27.6/29 26.4/28 25.9/27 |
| 13 | 3 ms | 34.6/39 39.3/46 40.4/54 35.1/52 31.6/33 26.7/28 25.5/42 25.0/31 24.9/28 28.4/32 |
| 19 | 3 ms | 26.9/28 27.0/28 29.3/38 35.0/43 40.2/78 27.2/33 25.9/31 28.6/32 25.3/27 24.8/26 |

The slow stretches rise and fall over hundreds of milliseconds and include many
calls. They are not single calls of several hundred milliseconds.

## Correctness

The correctness phase passed its usual gate: minimum cosine 0.9991391 against the
CPU fp32 reference, and repeated vectors byte-identical (`report.json`). Every one
of the 10,000 timed predictions, in both arms and for all 8 rows, was compared bit
for bit with its row's correctness-phase vector. There were 0 mismatches
(`identical` column of `per-call.csv`). The gap does not change outputs.

## Conditions and caveats

- The run started only once the 5-minute load average was below 12 (10.35) and
  `campaign_rig_claim` in the prefrontal store was empty. Load then rose from
  other work on the machine: the 5-minute average was 21.0 when the run ended
  (`monitor.log`, sampled every 15 s). So most blocks ran above the intended
  ceiling. The arms were interleaved so that both experienced the same drift.
- The rig claim stayed empty throughout. No campaign or other direct-ANE worker
  was running.
- Production Synapse logged no lines between 12:05:48Z and 12:10:49Z
  (`synapse-during-run.log` is empty), so it served no requests on any lane
  during the run. It had restarted at 11:41:55Z. Its recent jobs ran on the
  metal and decode lanes, not on the ANE.
- Timing is wall clock. `hw_execution_time_ns` still reads 0 on this hardware,
  so device time cannot be separated from dispatch.
- One run of 20 blocks. A negative at this scale does not rule out a rare stall
  that occurs less than once in 130,000 dispatches.

## Files

- `per-call.csv`: one row per timed prediction: block, arm, row ID, start time
  relative to the run, wall ms, measured sleep, byte-identity flag, and the 13
  per-dispatch times (separated by `;`).
- `summary.json`: the per-arm and per-block statistics above.
- `report.json`: the correctness report, including per-block load and each
  correctness-phase warm call.
- `monitor.log`, `run.log`, `synapse-during-run.log`: load and rig-claim samples,
  harness output, and production log lines in the run window.
- `series.svg`: the per-call series of every block.

## Command

From `bench/spikes/ane-direct-probe`, after
`env -u TMPDIR cargo build --release --bin modernbert_full`:

```sh
MODEL_SNAPSHOT="$HF_HOME/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots/e7f32e3c00f91d699e8c43b53106206bcc72bb22" \
  ./gap_ab.sh "$OUT" 0,3 20 500
python3 gap_analysis.py "$OUT/per-call.json" --svg "$OUT/series.svg" \
  --csv "$OUT/per-call.csv" --summary-json "$OUT/summary.json"
```

`gap_ab.sh` runs `modernbert_full "$MODEL_SNAPSHOT" rows.jsonl --seq 512
--layers-per-executable 1 --warm-repetitions 5 --block-gaps-ms 0,3 --blocks 20
--calls-per-block 500 --report … --per-call-out …`. The default `--gap-ms 0` and
`--blocks 0` leave the binary's existing behaviour unchanged.
