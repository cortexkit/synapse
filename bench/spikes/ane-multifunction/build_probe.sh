#!/usr/bin/env bash
# Build the Swift Core ML probe into the spike's artifact directory.
set -euo pipefail
root="${1:-$HOME/.local/share/cortexkit/synapse/ane-multifunction}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$root/bin"
swiftc -O -parse-as-library "$here/probe.swift" -o "$root/bin/probe"
echo "$root/bin/probe"
