#!/bin/sh
# C compiler shim for checking the Linux target from a Mac (see the Linux
# clippy block in scripts/train-push.local.sh). Crates with C build scripts
# (ring) call the target's C compiler through cc-rs, which passes
# --target=x86_64-unknown-linux-gnu. zig cannot parse the "unknown" vendor
# field and fails, so drop that flag and name zig's own spelling of the target.
for arg do
  shift
  case "$arg" in --target=*) continue ;; esac
  set -- "$@" "$arg"
done
exec zig cc -target x86_64-linux-gnu "$@"
