# Second NVIDIA progress-liveness rental

## Verification plan

Candidate: `be6cd118a40cc8cc5e999fad9e1e880e43dcd644`. Baseline:
`5f6d7419fc42a5ebf6ed1b8eab2115b6cc84fe5a`. The candidate's
`crates/synapse-engine-cuda` tree is identical to the baseline (empty
`git diff 5f6d7419 be6cd118 -- crates/synapse-engine-cuda`).

The recipe comes from `db3b0176cf619f5f45c9e447dc74cea4ccc9b117`, under
`docs/evidence/worker-progress-liveness/cuda-rental/`. The retained tools are
copied from that revision with only the candidate SHA, rental instance ID,
previously required X/GLVND dependencies and Mesa ICD removal, and separate
Cargo targets per revision made explicit. The redundant rebuild script is
omitted. The timing probe and mirror are unchanged. No tolerance, request
budget, timeout, or certification configuration is changed.

One RTX 5090 rental will use `nvidia/cuda:12.8.1-base-ubuntu24.04`, 100 GB
local disk, all NVIDIA driver capabilities, checksum-pinned CUDA 13.2.1
redistributables and Vulkan SDK 1.4.357.0. Eight candidate certification cells
and paired baseline/candidate CUDA timings will use the original recipe and
byte-pinned mirror. The baseline will also be built on the rental, even if its
GPU and driver match the previous rental.

The hard spend cap is $3.00. A local watchdog will destroy the instance at two
hours or a $2.00 observed account debit, leaving headroom for delayed billing
and bandwidth. Cleanup must confirm the account instance list contains none
created by this run. Raw logs and source bundles stay outside git under
`evidence/cuda-rental-2/`; their hashes and compact results will be committed
here. Initial account resources: no instances.

Status: planned; no GPU result is claimed yet.
