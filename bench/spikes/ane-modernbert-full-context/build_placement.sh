#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$SCRIPT_DIR/.build"
xcrun swiftc -O -target "${MACOS_TARGET:-arm64-apple-macos14.4}" -parse-as-library \
  "$SCRIPT_DIR/placement.swift" -o "$SCRIPT_DIR/.build/modernbert-placement"
echo "built $SCRIPT_DIR/.build/modernbert-placement"
