# Rental lifecycle

All amounts are USD; hard cap: $3.00. No pre-existing account instances.

| Instance | Offer / machine | GPU / driver | Compute/hour | 100 GB disk/hour | State |
| --- | --- | --- | ---: | ---: | --- |
| 55712356 | 53119697 / 145814 | RTX 5090 / 595.84 | 0.5133333333333333 | 0.0277777777777778 | created |

Created with Vast CLI 1.2.1 using the previous rental's image and options:

```sh
vastai create instance 53119697 --image nvidia/cuda:12.8.1-base-ubuntu24.04 \
  --disk 100 --ssh --direct --env '-e NVIDIA_DRIVER_CAPABILITIES=all' \
  --label mason-cuda-liveness-2-bg_6ac235e863cccc82 --cancel-unavail --raw \
  --onstart-cmd 'chown root:root /root /root/.ssh /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; echo SSH_KEY_MODES_REPAIRED'
```

A local watchdog destroys this instance at two hours or $2.00 observed account
credit drop, whichever occurs first. Two hours of compute plus storage cost
$1.0822222222222222. Advertised ingress is $2.6666666666666665/TB and egress
$4.00/TB. No persistent volume is created. Initial account credit:
$8.968145295700594. Final debit and destruction confirmation are pending.
