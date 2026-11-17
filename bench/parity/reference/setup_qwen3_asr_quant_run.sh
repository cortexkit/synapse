#!/usr/bin/env bash
# Prepare public inputs only. Never pass a private recording/list directory here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
WORK="${1:-$ROOT/quant-work}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
STARTED="$(date +%s)"
if [[ ! -f "$WORK/rental-start" ]]; then
  printf '%s\n' "${RENTAL_START:-$STARTED}" >"$WORK/rental-start"
fi
export PATH="$HOME/.cargo/bin:$PATH"
if [[ "$(uname -s)" != Linux ]] || ! command -v apt-get >/dev/null; then
  echo 'This setup script requires Ubuntu Linux.' >&2
  exit 1
fi
SUDO=()
if [[ "$(id -u)" != 0 ]]; then SUDO=(sudo); fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y python3 python3-venv python3-pip build-essential curl pkg-config libssl-dev libsndfile1
python3 bench/parity/reference/qwen3_asr_quant_run.py verify-kit
if ! command -v rustup >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$WORK/rustup-init.sh"
  sh "$WORK/rustup-init.sh" -y --profile minimal --default-toolchain 1.100.0
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustup toolchain install 1.100.0 --profile minimal
cargo +1.100.0 --version
rustc +1.100.0 --version
export CARGO_TARGET_DIR="$WORK/rust-target"
cargo +1.100.0 build --locked --release --manifest-path bench/parity/Cargo.toml --no-default-features --features q8-derive --bin q8-derive --bin qwen-asr-wer
mkdir -p "$WORK/bin"
cp "$CARGO_TARGET_DIR/release/q8-derive" "$WORK/bin/q8-derive"
cp "$CARGO_TARGET_DIR/release/qwen-asr-wer" "$WORK/bin/qwen-asr-wer"
python3 -m venv "$WORK/venv"
PY="$WORK/venv/bin/python"
"$PY" -m pip install --disable-pip-version-check pip==25.3
"$PY" -m pip install --disable-pip-version-check torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install --disable-pip-version-check -r bench/parity/reference/requirements-qwen3-asr.txt
# A setup retry may reopen its own cache; never run setup beside active workers.
if [[ -d "$WORK/cache" ]]; then chmod -R u+w "$WORK/cache"; fi
"$PY" bench/parity/reference/qwen3_asr_quant_run.py prepare --cache "$WORK/cache" --derive-binary "$WORK/bin/q8-derive" --setup-started-at "$STARTED"
echo "Prepared read-only public cache. WORK=$WORK"
echo "Start/resume commands: bench/parity/reference/QUANT-GATE-RUN.md"
