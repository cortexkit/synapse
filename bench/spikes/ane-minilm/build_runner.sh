#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="$SCRIPT_DIR/.build"
OUT_BIN="$OUT_DIR/ane-coreml"
TARGET_TRIPLE=${MACOS_TARGET:-arm64-apple-macos14.4}

mkdir -p "$OUT_DIR"

swiftc \
  -O \
  -target "$TARGET_TRIPLE" \
  -parse-as-library \
  "$SCRIPT_DIR/ane_coreml.swift" \
  -o "$OUT_BIN"

echo "built $OUT_BIN"
