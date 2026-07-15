#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="$SCRIPT_DIR/.build"
OUT_BIN="$OUT_DIR/ane-lfm2"
TARGET_TRIPLE=${MACOS_TARGET:-arm64-apple-macos15.0}

mkdir -p "$OUT_DIR"
swiftc -O -target "$TARGET_TRIPLE" -parse-as-library "$SCRIPT_DIR/ane_lfm2.swift" -o "$OUT_BIN"
echo "built $OUT_BIN"
