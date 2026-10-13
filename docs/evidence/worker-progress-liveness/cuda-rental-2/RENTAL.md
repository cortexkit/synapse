# Rental lifecycle

All amounts are USD; hard cap: $3.00. No pre-existing account instances.

| Instance | Offer / machine | GPU / driver | Compute/hour | 100 GB disk/hour | State |
| --- | --- | --- | ---: | ---: | --- |
| 55712356 | 53119697 / 145814 | RTX 5090 / 595.84 | 0.5133333333333333 | 0.0277777777777778 | destroyed |

Created with Vast CLI 1.2.1 using the previous rental's image and options:

```sh
vastai create instance 53119697 --image nvidia/cuda:12.8.1-base-ubuntu24.04 \
  --disk 100 --ssh --direct --env '-e NVIDIA_DRIVER_CAPABILITIES=all' \
  --label synapse-cuda-liveness-2 --cancel-unavail --raw \
  --onstart-cmd 'chown root:root /root /root/.ssh /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; echo SSH_KEY_MODES_REPAIRED'
```

A local watchdog destroys this instance at two hours or $2.00 observed account
credit drop, whichever occurs first. Two hours of compute plus storage cost
$1.0822222222222222. Advertised ingress is $2.6666666666666665/TB and egress
$4.00/TB. No persistent volume is created. Initial account credit:
$8.968145295700594. The final checked credit is $8.667933213700621:
**$0.3002120819999732 account debit**, comfortably below the $3.00 cap.

| Instance | Created UTC | Destruction confirmed UTC | Final billing checked UTC | Rounded instance invoice |
| --- | --- | --- | --- | ---: |
| 55712356 | 2026-10-13T08:48:12Z | 2026-10-13T09:17:10Z | 2026-10-13T09:22:52Z | $0.300 |

The immediate post-destruction credit was $8.705675459700643; an additional
$0.03774224600002185 posted afterward. The final invoice includes $0.272 compute,
$0.015 disk and $0.013 download, totaling $0.300 after rounding. Account debit is
reported separately at full precision, not inferred from the hourly rate or
rounded invoice. Billing increments exceed the wall-clock runtime.

Cleanup used `vastai destroy instance 55712356 --yes --raw` (exit 0); its stdout
was empty. Successful destruction was independently established by
`vastai show instances --raw` returning `[]` immediately afterward and again
at the final billing check. `vastai show volumes --raw` also returned `[]`.
The local watchdog exited normally after explicit destruction; it did not reach
its time or debit trigger. No other instance or volume was created.

Raw account responses, lifecycle output and watchdog logs stay outside git;
[raw-files.sha256](raw-files.sha256) retains their hashes. Only sanitized billing
and resource metadata is committed in [spend.json](spend.json).
