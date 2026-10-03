#!/bin/sh
# Archiver paired with zig-cc-linux.sh: cc-rs runs AR as a single program
# path, so "zig ar" needs a wrapper of its own.
exec zig ar "$@"
