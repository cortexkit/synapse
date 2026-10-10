# vast.ai rentals for the CUDA and Vulkan catalog lane proof

## Rental 2 (the machine used for the evidence)

| field | value |
| --- | --- |
| instance id | 55175162 |
| offer id | 53288946 (machine 152326, Nevada, US) |
| GPU | 1x RTX 5070 Ti, 16 GB (compute capability 12.0) |
| NVIDIA driver | 610.57.04 (vast `cuda_max_good` 13.3) |
| CPU | AMD Ryzen 9 5900XT, 32 effective vCPUs, 62 GB RAM |
| image | `nvidia/cuda:12.8.1-base-ubuntu24.04`, `NVIDIA_DRIVER_CAPABILITIES=all` |
| disk | 120 GB |
| price | $0.316/h total (GPU offer $0.294/h plus disk) |
| created | 2026-10-10T08:42:59Z |

## Rental 1 (destroyed unused)

| field | value |
| --- | --- |
| instance id | 55174610 |
| offer id | 47783570 (machine 33558, Quebec, CA) |
| GPU | 1x RTX 4080 SUPER (compute capability 8.9) |
| NVIDIA driver | 595.84 (vast `cuda_max_good` 13.2) |
| price | $0.329/h total |
| created | 2026-10-10T08:38:20Z |
| destroyed | about 2026-10-10T08:42Z |
| why | sshd refused every login: the host's `/root/.ssh/authorized_keys` had "bad ownership or modes" (from `vastai logs`), so the box was unreachable |
