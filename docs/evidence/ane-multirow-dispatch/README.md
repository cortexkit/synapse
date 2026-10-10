# Direct Neural Engine lane: several rows per pass

Measured on the Mac mini (Apple M4, 16 GB, macOS build 26A428), release build,
one process per arm. Production Synapse on the development Mac was not touched:
no Neural Engine compile or inference ran there. Raw JSON is in `raw/`; it holds
hashes, token counts and fixture case names, never input text.

**Follow-up:** [whole-export full-planner and runtime tight-packing results](../ane-tight-packing/README.md)
measure both 64-row and production eight-row planning windows. They replace the
~82 s estimate for tightly filling 256-column variable-length passes below with
measured whole-export walls of 101 s (64-row planning) and 116 s (eight-row).

## Conclusions, ranked by confidence

1. **High: several rows can share one pass, and the outputs do not change.**
   The width-folded layout (rows side by side on the width axis, attention
   folded onto the channel axis next to the heads) returned vectors
   **bit-identical** to the single-row path for every row checked: 19 to 43 rows
   per arm, rows of 4 to 128 tokens mixed in one pass, all 64 rows of the
   export's first batch, and all **6,341** rows of the whole export (twice).
   Empty slots and neighbouring rows never changed a real row (max abs
   difference 0.0), and a row returned the same vector from the first and the
   last slot. The single-row path still meets the fp32 gate (minimum cosine
   0.99988 over the fixture cases that fit, gate 0.999).
2. **High: the cost of a pass follows its total width, not its row count.**
   Per-pass wall time is about 18 ms for 64 columns, 21 ms for 128, 30–32 ms for
   256, 58 ms for 384, 77–83 ms for 512 and 165–176 ms for 1,024, whatever the
   split into rows. 256 columns is the cheapest per column (0.12 ms against
   0.16 ms at 128 and 0.15–0.16 ms at 512 and above). So the weight-read
   hypothesis holds only in part: a pass has a large fixed cost (roughly 15 ms
   of the 21 ms at 128 columns, from the 64/128/256-column points), which
   sharing amortises, but above 256 columns the per-column cost rises again and
   swallows the gain. The cause of that rise is not measured; on-chip memory
   pressure from the 3,072-wide MLP activations is a guess, not a finding.
3. **High: the gain is real but modest for N rows of 128, large for short rows.**
   Two 128-token rows per pass: **1.31×** per row (21.0 → 16.0 ms). Three or more
   128-token rows per pass, or any multi-row pass at 256, is no faster or slower
   than one row at a time. Rows much shorter than 128 tokens gain a lot from
   narrow slots, because today's path pads every row to 128: 4 rows in 64-token
   slots **2.65×** (7.9 ms per row), 8 rows in 32-token slots **5.4×** (3.8 ms).
4. **High: the first real 64-row batch runs 2.9× faster; the whole export 1.2×.**
   First batch (60 of its 64 rows are at most 64 tokens): 1.33 s → 0.46 s
   (48 → 139 rows/s) with 8×32 and 4×64 programs; 1.33 s → 0.56 s with 4×64
   alone; 1.35 s → 1.04 s with 2×128. Whole export (6,341 rows, mean 107
   tokens, 22 % of rows above 128 tokens): 163 s → 132 s and, in an earlier
   run, 155 s → 134 s with 2×128, so **1.16–1.23×** (41 → 47–48 rows/s). The
   first batch is far shorter than the export as a whole; do not extrapolate
   its speed-up.
5. **High: the batch axis compiles and runs but silently corrupts rows.**
   With 3, 4 or 8 rows of width 128 on the NCHW batch axis, rows in slots 2 and 3
   (and 6 and 7 of 8) come back wrong (cosine 0.73–0.92 against the same row
   alone) with no error. Two rows, and 2 or 4 rows of width 256, matched to
   cosine 0.99999, but not bit-identically. That layout is a failure; the
   prototype refuses it.
6. **High: the block-diagonal-mask arm is correct but slower.** Packing rows on
   width and masking across rows wastes attention work: 1.16× at 2×128 and
   0.63–0.96× everywhere else. The segmented arm (attention sliced per row) is
   correct and close to width-folded but always slightly slower.
7. **Medium (estimate, not measured): packing rows of different lengths tightly
   into about 256 columns is the next lever.** Today's 155–163 s for the export
   is 679,090 real tokens in 6,341 passes padded to 128 or 256. Filled 256-column
   passes at the measured ~31 ms would need about 2,650 passes, about 82 s. That
   needs per-pass masks and rotary positions as program inputs, which no arm
   here tested.

## What was tested

Code: `crates/synapse-worker-ane-direct/src/multirow.rs`. A `MultiRowShape` is a
layout, a row count N and a slot width; each shape compiles 28 per-layer
programs. Every layout computes the same Qwen3 layer as `qwen.rs` per row.
Qwen3 attention is causal and rows are right-padded, so real tokens never see
padding; the multi-row programs take only the activation tensor (no mask
input). Rotary positions restart at zero in every slot.

| Layout | Activation tensor | Attention |
| --- | --- | --- |
| `batch-axis` | `[N, 1024, 1, W]` | batched over N, per 128-query tile |
| `width-segmented` | `[1, 1024, 1, N·W]` | each row's segment sliced out and run alone |
| `width-folded` | `[1, 1024, 1, N·W]` | reshaped to `heads·N` channels of width W, one batched multiply |
| `width-block-mask` | `[1, 1024, 1, N·W]` | full packed width, block-diagonal causal mask |

The binding accepts all four. Every shape tried compiled (28 of 28
executables); none was refused and none crashed. Slot widths below 128 are
allowed while the tensor stays at least 64 columns wide (the binding's
`MIN_SPATIAL_WIDTH`); 16-token slots were not measured because too few real rows
fit them (the export has no row of 16 tokens or fewer).

### Method

One process per arm (`run-arm.sh`): load the converted package, admit the
single-row rungs up to the slot width (28 executables each), compile the
multi-row program (28 more), then:

1. **Correctness first.** Every fixture case that fits the slot runs alone and is
   compared with its fp32 reference from `bench/parity` (gate 0.999). The same
   fixture rows plus the first 4·N export rows that fit run in multi-row passes,
   N at a time, mixed lengths in one pass; each slot is compared with the same
   row run alone (gate 0.9999, same 1,024 dimensions). Then one row runs with
   the other slots empty, surrounded by other rows, and in the last slot.
2. **Timing.** The first N export rows that fill the rung (for width 256: rows of
   129–256 tokens, so the single-row path also runs at 256) run as N single-row
   calls and as one multi-row pass, after one warm-up of each, for 5
   repetitions alternating which path goes first. One-minute load average is
   read before and after every sample. Compile time is reported separately.
3. **Replay.** The export's first batch (rows 0–63), and with
   `ANE_MULTIROW_REPLAY_ALL` all 127 batches of the export's own plan, run
   today's way (each row alone at its rung, in order) and with multi-row passes
   planned within each batch (each row in the narrowest slot that fits, longer
   rows alone), alternating, 5 repetitions for the first batch and 3 for the
   whole export. Every output is compared with today's path first.

Tokenization is the catalog's Qwen3 document composition (tokenizer special
tokens, then one terminal end-of-text token); it reproduced the fixture token
ids for all 17 fixture cases.

The mini's one-minute load was 2–5 during the runs (system processes such as
`fileproviderd` and an open GUI session; nothing of ours ran alongside). One
whole-export single-row sample ran while load reached 9.05 and is slower
(186 s); the medians use all samples.

## Results per arm (final run)

Per-row time is the median of 5 alternating repetitions. Speed-up is single-row
per-row time over multi-row per-row time. Parity is the minimum cosine of any
multi-row slot against the same row alone; "identical" counts bit-identical
rows. Compile seconds are for the 28 multi-row executables; a single-row rung
compiled in 10–28 s at 128 and 14–21 s at 256 in the same processes.

### 128-token slots

| Layout | N | ms/row single | ms/row multi | Speed-up | Compile s | Exe | Parity min cos | Identical | Leak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| width-folded | 2 | 21.43 | 16.15 | **1.33** | 31.3 | 28 | 1.0000000 | 19/19 | none |
| width-folded | 3 | 20.97 | 19.36 | 1.08 | 19.5 | 28 | 1.0000000 | 23/23 | none |
| width-folded | 4 | 20.91 | 19.63 | 1.07 | 17.0 | 28 | 1.0000000 | 27/27 | none |
| width-folded | 8 | 21.28 | 20.63 | 1.03 | 18.8 | 28 | 1.0000000 | 43/43 | none |
| width-segmented | 2 | 21.20 | 17.19 | 1.23 | 13.2 | 28 | 1.0000000 | 19/19 | none |
| width-segmented | 3 | 20.96 | 19.90 | 1.05 | 17.6 | 28 | 1.0000000 | 23/23 | none |
| width-segmented | 4 | 21.16 | 21.41 | 0.99 | 16.4 | 28 | 1.0000000 | 27/27 | none |
| width-segmented | 8 | 20.81 | 21.74 | 0.96 | 21.0 | 28 | 1.0000000 | 43/43 | none |
| width-block-mask | 2 | 20.44 | 17.63 | 1.16 | 13.6 | 28 | 1.0000000 | 19/19 | none |
| width-block-mask | 3 | 20.71 | 21.57 | 0.96 | 19.7 | 28 | 1.0000000 | 23/23 | none |
| width-block-mask | 4 | 20.61 | 24.72 | 0.83 | 20.2 | 28 | 1.0000000 | 27/27 | none |
| width-block-mask | 8 | 21.40 | 29.16 | 0.73 | 25.7 | 28 | 1.0000000 | 43/43 | none |
| batch-axis | 2 | 21.05 | 16.11 | 1.31 | 16.9 | 28 | 0.9999937 | 15/19 | none |
| batch-axis | 3 | 20.93 | 17.68 | *invalid* | 19.8 | 28 | **0.7562** | 12/23 | **slot 2 wrong** |
| batch-axis | 4 | 20.96 | 19.05 | *invalid* | 18.9 | 28 | **0.8021** | 11/27 | **slots 2–3 wrong** |
| batch-axis | 8 | 20.97 | 19.06 | *invalid* | 25.3 | 28 | **0.8693** | 19/43 | **slots 2, 3, 6, 7 wrong** |

### 256-token slots (single-row baseline also at 256)

| Layout | N | ms/row single | ms/row multi | Speed-up | Compile s | Exe | Parity min cos | Identical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| width-folded | 2 | 36.46 | 41.40 | 0.88 | 20.2 | 28 | 1.0000000 | 21/21 |
| width-folded | 4 | 37.81 | 44.10 | 0.86 | 20.8 | 28 | 1.0000000 | 29/29 |
| width-segmented | 2 | 36.93 | 44.46 | 0.83 | 20.4 | 28 | 1.0000000 | 21/21 |
| width-segmented | 4 | 36.67 | 45.36 | 0.81 | 25.4 | 28 | 1.0000000 | 29/29 |
| width-block-mask | 2 | 36.58 | 49.21 | 0.74 | 22.3 | 28 | 1.0000000 | 21/21 |
| width-block-mask | 4 | 36.53 | 58.18 | 0.63 | 29.6 | 28 | 1.0000000 | 29/29 |
| batch-axis | 2 | 37.52 | 44.48 | 0.84 | 20.8 | 28 | 0.9999937 | 15/21 |
| batch-axis | 4 | 37.11 | 43.78 | 0.85 | 29.7 | 28 | 0.9999937 | 21/29 |

### Narrow slots (single-row baseline at 128, today's smallest rung)

| Layout | N × slot | Columns | ms/row single | ms/row multi | Speed-up | Compile s | Exe | Parity min cos | Identical |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| width-folded | 2 × 64 | 128 | 20.72 | 10.57 | 1.96 | 14.0 | 28 | 1.0000000 | 16/16 |
| width-folded | 4 × 64 | 256 | 21.28 | 8.09 | **2.63** | 17.6 | 28 | 1.0000000 | 24/24 |
| width-folded | 8 × 64 | 512 | 20.67 | 9.67 | 2.14 | 18.8 | 28 | 1.0000000 | 40/40 |
| width-folded | 2 × 32 | 64 | 20.31 | 8.95 | 2.27 | 13.4 | 28 | 1.0000000 | 14/14 |
| width-folded | 4 × 32 | 128 | 20.57 | 5.05 | 4.07 | 12.8 | 28 | 1.0000000 | 22/22 |
| width-folded | 8 × 32 | 256 | 20.64 | 3.79 | **5.44** | 15.7 | 28 | 1.0000000 | 38/38 |
| width-segmented | 2 × 64 | 128 | 21.03 | 10.98 | 1.91 | 16.2 | 28 | 1.0000000 | 16/16 |
| width-segmented | 4 × 64 | 256 | 20.94 | 8.32 | 2.52 | 17.9 | 28 | 1.0000000 | 24/24 |
| width-segmented | 8 × 64 | 512 | 20.91 | 10.04 | 2.08 | 21.8 | 28 | 1.0000000 | 40/40 |
| width-segmented | 2 × 32 | 64 | 20.40 | 9.10 | 2.24 | 16.9 | 28 | 1.0000000 | 14/14 |
| width-segmented | 4 × 32 | 128 | 20.97 | 5.45 | 3.85 | 13.7 | 28 | 1.0000000 | 22/22 |
| width-segmented | 8 × 32 | 256 | 20.73 | 4.05 | 5.12 | 23.9 | 28 | 1.0000000 | 38/38 |
| batch-axis | 2 × 64 | 64 | 20.73 | 10.79 | 1.92 | 14.6 | 28 | 0.9999945 | 15/16 |

Leak checks for every width-* arm: a row alone with empty slots and the same
row among neighbours were bit-identical (max abs 0.0), and first slot against
last slot was bit-identical. Per-arm load ranges are in `summary.json`.

The exploratory runs in `raw/exploratory/` (earlier builds of the same harness
during development, same timing and parity code) reproduce these numbers within
about 5 %: width-folded 2×128 1.27–1.31× over four runs, 8×32 5.36–5.49×, 4×64
2.59–2.66×, and the same batch-axis failures in the same slots (minimum cosine
0.73 for 3 rows).

## 64-row replay (export batch 0, rows 0–63)

The first batch is short: 60 of its 64 rows have at most 64 tokens (19–101
tokens, 2,489 in total), so all rows fit a 128 slot.

| Path | Programs | Executables held | Median wall | Rows/s | Walls (ms) | Min cos vs today | Identical | Load |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| today (one row at a time) | 128 rung | — | 1.35 s | 47.4 | 1351, 1335, 1321, 1358, 1360 | — | — | 3.62–3.76 |
| multi-row | width-folded 2×128 | 84 | 1.04 s | 61.8 | 1036, 1037, 1035, 1031, 1045 | 1.0000000 | 64/64 | 3.62–3.68 |
| today | 128 rung | — | 1.33 s | 48.0 | 1329, 1329, 1340, 1332, 1339 | — | — | 2.29–2.53 |
| multi-row | width-folded 4×64 (+4 rows alone) | 56 | 0.56 s | 115.3 | 549, 555, 558, 553, 558 | 1.0000000 | 64/64 | 2.29–2.53 |
| today | 128 rung | — | 1.33 s | 47.9 | 1331, 1347, 1334, 1337, 1335 | — | — | 2.62–2.67 |
| multi-row | width-folded 8×32 + 4×64 (+4 rows alone) | 84 | **0.46 s** | **138.7** | 461, 462, 455, 461, 463 | 1.0000000 | 64/64 | 2.62–2.81 |

Each pair ran in one process, alternating; "today" rows are the single-row path
in the same process. The 2×128 process also held the 256 rung for the
whole-export replay, hence 84 executables.

## Whole-export replay (127 batches, 6,341 rows)

Width-folded 2×128 for rows of at most 128 tokens (2,496 passes), every longer
row alone at 256 (1,403 rows); single-row 128 and 256 rungs plus the program,
84 executables. Three alternating repetitions.

| Run | Path | Median wall | Rows/s | Walls (s) | Min cos | Identical | Load |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| final | today | 163.1 s | 38.9 | 163.1, 160.4, 186.0 | — | — | 2.02–9.05 |
| final | multi-row | 132.2 s | 48.0 | 132.2, 131.8, 133.9 | 1.0000000 | 6341/6341 | 2.02–3.63 |
| exploratory | today | 155.2 s | 40.9 | 155.7, 155.2, 154.6 | — | — | 1.91–2.40 |
| exploratory | multi-row | 133.8 s | 47.4 | 132.0, 134.4, 133.8 | 1.0000000 | 6341/6341 | 1.91–2.67 |

The export's composed lengths: 152 rows of at most 32 tokens, 653 of at most
64, 4,938 of at most 128, all 6,341 of at most 256; 679,090 tokens in total.
Adding 8×32 and 4×64 programs for the short rows would, from the per-row costs
above, bring the export to about 125 s (an estimate), but three programs plus
two rungs is 140 executables, above one process's capacity.

## Executable cost

Every multi-row shape is 28 more executables (one per layer), compiled in 13–31 s
on the mini. One process holds roughly 115 and the supervisor budgets 100; a
lane that keeps the 128 and 256 rungs (56) has room for one multi-row shape
before it must evict. The prototype leaves budgeting to its caller.

## Prototype

`crates/synapse-worker-ane-direct/src/multirow.rs`, opt-in only:
`Model::compile_multirow(MultiRowShape)` builds a `MultiRowProgram`;
`MultiRowProgram::run(&model, rows)` embeds up to N rows that fit a slot and
returns one vector per row in input order; `plan_passes(lengths, shapes)` puts
each row of a call in the narrowest fitting slot and leaves longer rows for the
single-row path. Only Qwen3 embedding profiles are accepted; the batch-axis
layout is refused (the hardware experiment compiles it through a crate-private
path). Compile, run and drop each happen inside an autorelease pool. Nothing in
the request loop or `synapse-module` calls it; serving is a later task.

The single-row path is unchanged: `qwen.rs` only exposes `rms` to the new
module, and `single_row_layer_program_is_unchanged` pins a canonical digest of
the single-row layer program (weight references replaced by the SHA-256 of the
bytes they point at, lines sorted, because the binding emits constants in
hash-map order). Changing the single-row causal mask value from -10,000 to
-9,000 turned exactly that test red.

Unit tests (no Neural Engine): shape validation, packed index layout, rotary
positions restarting per slot, the causal and cross-row attention bias, pass
planning, slot packing and padding, request refusals, the embedding tail, graph
construction for every layout, and the single-row digest.

## Unresolved

- Why the batch axis corrupts slots 2 and 3 (and 6 and 7) at width 128 but not
  at width 256 was not investigated. A guess: a batched operation tiles the
  batch in pairs at that width. Do not use the batch axis on this binding.
- Why per-column cost rises above 256 columns. Hardware statistics read 0 on
  this M4, so the split between memory traffic and compute is inferred.
- Variable-length packing with runtime masks and rotary inputs (conclusion 7) is
  untested; its op count, compile cost and parity are unknown.
- Slot widths that are not powers of two (for example 96 or 192, for rows of
  65–256 tokens) were not tried; the prototype refuses them.
- Only one machine (M4, build 26A428) was measured; other chips and OS builds
  may differ, especially for the batch-axis failure.
- Concurrency with a second Neural Engine client and the module's IPC overhead
  were not measured; this harness calls the backend in-process.
- The mini's background load was 2–5 throughout; one whole-export baseline
  sample ran at load up to 9.

## Reproduce on the Mac mini

Scratch directory: `~/mason-ane-multirow` (worktree copy in `synapse/`, build in
`target/`, converted package in `packages/`, runs in `runs/`). The converted
package is `bench/parity/models.json`'s pinned Qwen3 package
(`sha256:1af6d091…bd5db`); this run copied the one a previous profiling run had
converted on the mini. From the development Mac, at this branch's checkout:

```sh
rsync -a --delete --exclude target --exclude .git ./ macmini:mason-ane-multirow/synapse/
scp docs/evidence/ane-multirow-dispatch/{run-arm.sh,run-final.sh,summarize.py} \
  macmini:mason-ane-multirow/
```

On the mini:

```sh
source ~/.cargo/env
cd ~/mason-ane-multirow/synapse
export CARGO_BUILD_JOBS=6 DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  CARGO_TARGET_DIR=$HOME/mason-ane-multirow/target
cargo test --release --locked -p synapse-worker-ane-direct --no-run
cd ~/mason-ane-multirow
mkdir -p packages && cp <converted package> packages/qwen3-embedding-0.6b.safetensors

# One arm (writes JSON; exits non-zero if a correctness gate fails):
./run-arm.sh width-folded:4:64 $HOME/mason-ane-multirow/runs/one.json
# One arm plus the first-batch replay, with an extra program for the replay:
./run-arm.sh width-folded:8:32 $HOME/mason-ane-multirow/runs/replay.json \
  ANE_MULTIROW_REPLAY=1 ANE_MULTIROW_REPLAY_EXTRA=width-folded:4:64
# The whole final evidence run (about 55 minutes):
./run-final.sh
python3 summarize.py runs/final/arms/*.json runs/final/replay/*.json --json summary.json
```

`run-arm.sh` runs the ignored test `multirow::hardware::multirow_experiment`
from the crate directory with `TMPDIR` unset. Its inputs are environment
variables documented on the test: `ANE_MULTIROW_ARM` (`layout:rows:width`;
setting it is what enables the experiment), `ANE_TEST_PACKAGES`,
`ANE_MULTIROW_INPUT` (the `engram.jsonl` export; its `.meta.json` must sit
beside it for `ANE_MULTIROW_REPLAY_ALL`), `ANE_MULTIROW_TOKENIZER` (the
checkpoint's `tokenizer.json`), `ANE_MULTIROW_OUT`, and optionally
`ANE_MULTIROW_REPEATS`, `ANE_MULTIROW_REPLAY`, `ANE_MULTIROW_REPLAY_EXTRA` and
`ANE_MULTIROW_REPLAY_ALL`. `run-matrix.sh` is the exploratory runner used for
`raw/exploratory/matrix-1`.

## Files

- `summary.json`: the tables above, generated by `summarize.py` from `raw/final`.
- `raw/final/arms/*.json`: one report per arm; `raw/final/replay/*.json`: the
  replay runs (arm data plus `replay`, `replay_setup` and `replay_all`).
- `raw/exploratory/`: development runs with earlier builds of the harness.
- `run-arm.sh`, `run-final.sh`, `run-matrix.sh`, `summarize.py`: the scripts as run.
