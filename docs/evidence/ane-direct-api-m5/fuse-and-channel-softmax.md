# Two-layer fusion at 512 and channel-axis softmax: paired re-measurement

2026-09-29, Apple M5 Max, macOS 27 (26A428), shared workstation.

A campaign run proposed two graph changes to `bench/spikes/ane-direct-probe`
that beat unchanged code in all five of their paired samples but did not clear
the campaign's promotion gate (lower 95% bound of +5%; both sat at +3.8%). This
record re-measures them separately and together, with more pairs, and is the
reason the combination was landed.

- **fuse**: at sequence 512, compile two encoder layers per executable instead
  of one. Sequences 1024 and 2048 keep the caller's grouping (one layer).
- **softmax**: in attention, transpose the scores `[1, heads, queries, keys]`
  to `[1, keys, heads, queries]`, take the softmax on axis 1, and transpose
  back before the value matmul, instead of a softmax on the last axis.
- **combined**: both.

## Method

Each variant is a release build of the probe at the same master commit plus
the change, against the same pinned ANE binding commit
(`ec54af9501d4bfd0cf3a4b162e59022dee2118cb`) and model snapshot
(`e7f32e3c00f91d699e8c43b53106206bcc72bb22`). Every run calls `modernbert_full`
exactly as `bench/campaign/ane-direct-embed-harness.sh` does: the protected
`rows.jsonl`, `--layers-per-executable 1`, `--warm-repetitions 7`, sequences
512, 1024 and 2048, `TMPDIR` unset. Each report goes through the harness's own
`validate_report` (loaded from the script), so the 0.999 cosine gate, the
byte-identical repeat, and the aggregate throughput
(`sum(active_tokens) * 1000 / sum(warm_wall_ms_median)`) are the harness's
code, not a copy.

One sample is a full 512 → 1024 → 2048 sequence for one variant. Ten rounds;
each round runs a control sample then a candidate sample for each of the three
candidates, with the candidate order rotated every round, so all three
candidates share one interleaved schedule and each pair is adjacent in time.
60 samples, 180 shape runs.

Before every shape run the driver checked that no campaign held
`campaign_rig_claim` (none did at any point), that no measurement lock or
`Runner.Worker` was present, and that the 1-minute load was under 16; otherwise
it waited. Every admitted run started under 16 (highest admission 15.999). The
box is shared, and load sometimes rose during a run, up to 77 at run end; each
row of the tables records 1/5/15-minute load at the start and end of the 512
runs and the highest 1-minute load seen across the pair's six shape runs.

Interval: median of per-pair ratios (candidate / control), bootstrap 95%
interval over 10,000 resamples of the ten ratios, seed 2609.

## Quality (every variant, two repeats per sequence)

| variant | min cosine 512 | 1024 | 2048 | byte-identical across repeats |
|---|---:|---:|---:|---|
| control | 0.9991073 | 0.9991632 | 0.9990908 | yes, all shapes |
| fuse | 0.9991073 | 0.9991632 | 0.9990908 | yes, all shapes |
| softmax | 0.99913913 | 0.99918234 | 0.9990761 | yes, all shapes |
| combined | 0.99913913 | 0.99918234 | 0.9990761 | yes, all shapes |

All four pass. Fusion returns vectors byte-identical to the unfused graph at
every shape (fuse equals control, combined equals softmax). The channel-axis
softmax changes the fp16 rounding: vectors differ from control, raising the
minimum cosine slightly at 512 and 1024 and lowering it by 1.5e-6 at 2048,
which is still above the gate.

## Results

Throughput is aggregate tok/s. Ratios above 1 mean the candidate was faster.

### fuse

| pair | 512 control | 512 candidate | 512 ratio | 1024 ratio | 2048 ratio | 512 control load 1/5/15 (start → end) | 512 candidate load 1/5/15 (start → end) | max 1-min load, any shape |
|---|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 9,203 | 9,612 | 1.044 | 1.001 | 1.000 | 7.4/12.2/11.7 → 7.9/12.1/11.7 | 7.7/11.0/11.4 → 7.1/10.7/11.2 | 11.0 |
| 2 | 8,409 | 9,631 | 1.145 | 0.968 | 0.958 | 6.0/5.8/7.3 → 8.6/6.4/7.5 | 6.5/8.3/8.3 → 6.2/8.1/8.2 | 28.8 |
| 3 | 5,838 | 6,976 | 1.195 | 0.928 | 1.174 | 15.9/23.6/22.9 → 20.9/23.7/23.0 | 15.5/29.9/37.1 → 14.6/28.8/36.5 | 77.4 |
| 4 | 3,192 | 7,998 | 2.506 | 0.920 | 0.913 | 12.7/17.6/25.0 → 32.9/22.4/26.3 | 11.0/19.9/24.6 → 11.9/19.7/24.4 | 32.9 |
| 5 | 8,707 | 7,821 | 0.898 | 0.780 | 0.781 | 5.8/14.3/23.7 → 7.3/13.8/23.2 | 14.3/13.9/21.8 → 15.3/14.2/21.7 | 35.7 |
| 6 | 8,840 | 9,454 | 1.070 | 1.438 | 0.968 | 10.3/14.9/19.1 → 10.0/14.5/18.9 | 9.0/12.8/17.7 → 7.9/12.4/17.4 | 10.6 |
| 7 | 7,438 | 6,824 | 0.917 | 1.052 | 1.037 | 13.0/10.1/14.3 → 13.1/10.2/14.3 | 11.2/10.9/13.9 → 12.3/11.2/13.9 | 16.2 |
| 8 | 9,138 | 9,632 | 1.054 | 1.023 | 1.021 | 12.9/19.8/33.6 → 11.6/19.1/33.2 | 9.8/16.1/30.3 → 9.1/15.7/29.9 | 12.9 |
| 9 | 7,547 | 8,274 | 1.096 | 1.013 | 1.075 | 15.0/13.9/23.9 → 13.5/13.7/23.6 | 11.1/12.5/21.9 → 12.1/12.6/21.7 | 15.0 |
| 10 | 9,222 | 9,707 | 1.053 | 1.010 | 1.005 | 8.1/9.3/16.6 → 7.9/9.2/16.5 | 5.3/8.0/15.2 → 4.7/7.7/15.0 | 8.1 |

- 512: median ratio 1.062, 95% interval [0.985, 1.146], candidate faster in 8/10, median tok/s control 8,558 vs candidate 8,864
- 1024: median ratio 1.006, 95% interval [0.928, 1.032], candidate faster in 6/10, median tok/s control 7,263 vs candidate 7,262
- 2048: median ratio 1.003, 95% interval [0.941, 1.048], candidate faster in 6/10, median tok/s control 5,465 vs candidate 5,498
- 512 sensitivity, pairs whose both 512 runs ended under load 16 (n=8): median 1.053 [0.917, 1.096]

### softmax

| pair | 512 control | 512 candidate | 512 ratio | 1024 ratio | 2048 ratio | 512 control load 1/5/15 (start → end) | 512 candidate load 1/5/15 (start → end) | max 1-min load, any shape |
|---|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 9,246 | 10,091 | 1.091 | 1.031 | 1.017 | 8.3/9.8/10.8 → 12.4/10.7/11.1 | 8.0/10.1/10.8 → 7.1/9.8/10.7 | 13.1 |
| 2 | 9,233 | 9,972 | 1.080 | 1.035 | 1.007 | 5.3/6.3/8.6 → 5.4/6.2/8.6 | 6.2/6.4/8.3 → 6.4/6.4/8.3 | 6.4 |
| 3 | 7,231 | 6,667 | 0.922 | 0.604 | 0.994 | 15.2/25.1/31.7 → 13.9/24.1/31.2 | 13.9/19.4/27.8 → 14.8/19.1/27.4 | 28.5 |
| 4 | 9,070 | 7,191 | 0.793 | 0.912 | 0.829 | 13.3/18.3/22.2 → 11.6/17.7/21.9 | 11.7/15.1/20.2 → 18.3/16.3/20.5 | 34.1 |
| 5 | 6,770 | 7,248 | 1.071 | 1.291 | 0.647 | 14.2/24.2/36.8 → 14.7/23.5/36.2 | 11.1/18.5/31.8 → 14.4/18.6/31.5 | 50.8 |
| 6 | 8,666 | 10,016 | 1.156 | 0.855 | 0.923 | 6.7/10.5/16.0 → 7.4/10.4/15.9 | 6.0/9.1/14.8 → 6.4/9.0/14.7 | 13.0 |
| 7 | 7,424 | 6,929 | 0.933 | 0.808 | 0.969 | 13.8/12.7/14.1 → 13.4/12.7/14.0 | 9.3/11.3/13.3 → 23.4/14.6/14.4 | 23.4 |
| 8 | 8,847 | 9,500 | 1.074 | 0.943 | 1.468 | 10.7/13.4/16.2 → 10.3/13.2/16.0 | 14.1/55.5/52.6 → 13.8/52.7/51.7 | 17.3 |
| 9 | 9,311 | 7,622 | 0.819 | 0.979 | 1.297 | 7.6/11.2/20.1 → 7.1/11.0/19.8 | 8.8/9.9/18.2 → 9.1/9.9/18.0 | 9.1 |
| 10 | 8,395 | 10,015 | 1.193 | 1.608 | 1.093 | 5.2/7.1/13.9 → 5.6/7.1/13.8 | 14.0/14.1/15.4 → 12.5/13.8/15.3 | 19.7 |

- 512: median ratio 1.072, 95% interval [0.876, 1.118], candidate faster in 6/10, median tok/s control 8,756 vs candidate 8,561
- 1024: median ratio 0.961, 95% interval [0.855, 1.161], candidate faster in 4/10, median tok/s control 7,007 vs candidate 7,038
- 2048: median ratio 1.000, 95% interval [0.899, 1.157], candidate faster in 5/10, median tok/s control 5,158 vs candidate 5,231
- 512 sensitivity, pairs whose both 512 runs ended under load 16 (n=8): median 1.077 [0.922, 1.156]

### combined

| pair | 512 control | 512 candidate | 512 ratio | 1024 ratio | 2048 ratio | 512 control load 1/5/15 (start → end) | 512 candidate load 1/5/15 (start → end) | max 1-min load, any shape |
|---|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 9,389 | 10,438 | 1.112 | 0.971 | 0.913 | 4.7/8.1/10.0 → 4.2/7.9/9.8 | 3.7/6.7/9.2 → 3.9/6.6/9.1 | 5.3 |
| 2 | 9,333 | 10,696 | 1.146 | 1.016 | 1.004 | 4.9/5.9/7.9 → 4.5/5.8/7.8 | 5.3/5.6/7.5 → 5.7/5.7/7.5 | 6.6 |
| 3 | 9,068 | 7,648 | 0.843 | 0.756 | 0.889 | 12.8/13.6/10.7 → 11.6/13.3/10.6 | 14.5/23.3/20.8 → 14.6/22.7/20.7 | 25.4 |
| 4 | 4,751 | 9,212 | 1.939 | 1.147 | 1.157 | 14.7/47.0/68.9 → 27.5/45.5/67.1 | 12.5/32.6/42.5 → 15.4/31.9/42.0 | 27.5 |
| 5 | 8,505 | 10,324 | 1.214 | 1.032 | 1.023 | 15.2/31.5/33.9 → 14.3/30.2/33.4 | 10.3/19.3/26.8 → 10.1/18.7/26.4 | 16.3 |
| 6 | 6,187 | 10,327 | 1.669 | 1.006 | 1.179 | 12.8/21.6/22.7 → 12.1/20.8/22.4 | 14.4/17.7/20.9 → 14.7/17.6/20.8 | 17.5 |
| 7 | 9,223 | 9,902 | 1.074 | 1.202 | 1.022 | 13.2/20.1/19.0 → 11.9/19.5/18.8 | 7.9/15.3/17.2 → 7.0/14.7/17.0 | 13.2 |
| 8 | 7,889 | 10,442 | 1.324 | 1.861 | 0.919 | 12.0/38.8/46.3 → 12.0/36.3/45.1 | 14.0/24.6/37.5 → 13.1/23.9/37.0 | 26.0 |
| 9 | 9,219 | 9,993 | 1.084 | 1.018 | 0.998 | 8.0/13.3/27.3 → 8.1/13.1/27.0 | 7.6/11.5/24.7 → 7.4/11.2/24.4 | 15.0 |
| 10 | 9,131 | 10,609 | 1.162 | 1.032 | 0.972 | 5.8/11.0/14.1 → 5.8/10.8/13.9 | 4.8/9.0/12.8 → 5.0/8.8/12.7 | 8.2 |

- 512: median ratio 1.154, 95% interval [1.084, 1.441], candidate faster in 9/10, median tok/s control 9,100 vs candidate 10,325
- 1024: median ratio 1.025, 95% interval [0.994, 1.147], candidate faster in 8/10, median tok/s control 7,603 vs candidate 7,728
- 2048: median ratio 1.001, 95% interval [0.919, 1.089], candidate faster in 5/10, median tok/s control 5,565 vs candidate 5,463
- 512 sensitivity, pairs whose both 512 runs ended under load 16 (n=9): median 1.146 [1.074, 1.324]

## Verdict

| variant | 512 median [95%] | 1024 median [95%] | 2048 median [95%] | qualifies |
|---|---|---|---|---|
| fuse | 1.062 [0.985, 1.146] | 1.006 [0.928, 1.032] | 1.003 [0.941, 1.048] | no: 512 lower bound below 1 |
| softmax | 1.072 [0.876, 1.118] | 0.961 [0.855, 1.161] | 1.000 [0.899, 1.157] | no: 512 lower bound below 1 |
| combined | 1.154 [1.084, 1.441] | 1.025 [0.994, 1.147] | 1.001 [0.919, 1.089] | **yes** |

- **Combined** is the only variant whose 512 gain has a lower bound above zero
  (+8.4%), faster in 9 of 10 pairs; 1024 is neutral-to-positive (8 of 10
  faster) and 2048 is flat (median 1.001, 5 of 10). It was landed.
- **Each change alone** points the same way (medians +6.2% and +7.2%, close to
  the earlier campaign's +6.7% and +7.1%) but ten noisy pairs do not exclude
  zero, so neither would be landed on its own evidence.
- **Stacking**: the product of the two single medians is 1.062 × 1.072 = 1.138;
  the combined median is 1.154, close to the product and slightly above it.
  The single-change intervals are too wide to call it more than multiplicative.
- Restricting to pairs whose 512 runs both ended under load 16 does not change
  the picture (combined: n=9, median 1.146 [1.074, 1.324]).
- Wide upper bounds come from pairs where the control ran into a slow burst
  (for example combined pair 4, control 4,751 tok/s). The candidate values
  are tighter than the control values, as in the campaign run; the medians and
  the lower bounds, not the upper bounds, carry the conclusion.

## Caveat on the fusion override

The fusion change overrides the grouping inside `compile_model` at sequence
512, while the report still echoes the command-line `--layers-per-executable`
(1). The harness therefore accepts it as the one-layer protocol. The vectors are
byte-identical to the one-layer graph, so correctness is unaffected, but anyone
reading `layers_per_executable` in a 512 report should know that it is the
requested value, not the compiled grouping; the checkpoint labels follow the
compiled grouping.
