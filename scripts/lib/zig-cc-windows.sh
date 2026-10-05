#!/bin/sh
# C compiler shim for the Windows clippy preflight (scripts/train-push.local.sh).
# cc-rs passes --target=x86_64-pc-windows-gnu, which zig can't parse, so drop
# that flag and name zig's own spelling of the target.
for arg do
  shift
  case "$arg" in --target=*) continue ;; esac
  set -- "$@" "$arg"
done
exec zig cc -target x86_64-windows-gnu "$@"
