#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="$SCRIPT_DIR/.build"
OUT_BIN="$OUT_DIR/ane-prefill-runner"
TARGET_TRIPLE=${MACOS_TARGET:-arm64-apple-macos14.4}

mkdir -p "$OUT_DIR"
DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer} \
  xcrun swiftc \
  -O \
  -target "$TARGET_TRIPLE" \
  -parse-as-library \
  "$SCRIPT_DIR/ane_prefill_runner.swift" \
  -o "$OUT_BIN"

echo "built $OUT_BIN"
