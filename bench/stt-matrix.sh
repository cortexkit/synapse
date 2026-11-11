#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
wavs=${1:?usage: bench/stt-matrix.sh WAV_DIRECTORY NEW_OUTPUT_DIRECTORY}
out=${2:?supply a new result directory}
[[ ! -e "$out" ]] || { echo "result directory already exists: $out" >&2; exit 2; }
mkdir -p "$out"
case "$(uname -s)" in
  Darwin) lanes=${STT_LANES:-"transcribe fluid apple"}; backends=${STT_BACKENDS:-metal}; suffix= ;;
  MINGW*|MSYS*|CYGWIN*) lanes=${STT_LANES:-transcribe}; backends=${STT_BACKENDS:-"cuda vulkan"}; suffix=.exe ;;
  Linux) lanes=${STT_LANES:-transcribe}; backends=${STT_BACKENDS:-"cuda vulkan"}; suffix= ;;
  *) echo "unsupported OS" >&2; exit 2 ;;
esac
bench=${STT_BENCH:-target/release/stt-bench$suffix}
table=${STT_TABLE:-target/release/stt-table$suffix}
results=()
run() {
  local name=$1 worker=$2 model=$3 revision=$4 backend=$5
  "$bench" --wavs "$wavs" --worker "$worker" --model "$model" \
    --model-revision "$revision" --backend "$backend" --output "$out/$name.json"
  results+=("$out/$name.json")
}
for lane in $lanes; do
  case "$lane" in
    transcribe)
      : "${STT_GGUF_DIR:?set STT_GGUF_DIR to the three pinned GGUF files}"
      for backend in $backends; do
        worker="target/stt-$backend/release/stt-transcribe-cpp$suffix"
        run "transcribe-$backend-ultra-q8" "$worker" "$STT_GGUF_DIR/parakeet-ultra-Q8_0.gguf" a1539be1b849ea7463b88343e5277b373d0c4ea9 "$backend"
        run "transcribe-$backend-v3-q8" "$worker" "$STT_GGUF_DIR/parakeet-tdt-0.6b-v3-Q8_0.gguf" 85ac09ea12fc4b1112fa76810059364bc6adc9de "$backend"
        run "transcribe-$backend-v3-f16" "$worker" "$STT_GGUF_DIR/parakeet-tdt-0.6b-v3-F16.gguf" 85ac09ea12fc4b1112fa76810059364bc6adc9de "$backend"
      done
      ;;
    fluid)
      : "${STT_COREML_DIR:?set STT_COREML_DIR to the pinned CoreML snapshot}"
      for backend in ane-pal6 ane-int8; do
        run "fluid-$backend" target/stt-fluidaudio/release/stt-fluidaudio "$STT_COREML_DIR" 7dd20fe6b1797d35f5e3307e8b1732d9a178edfe "$backend"
      done
      ;;
    apple)
      run apple-speech target/stt-apple-speech/release/stt-apple-speech system OS-managed apple-speech
      ;;
    *) echo "unknown lane: $lane" >&2; exit 2 ;;
  esac
done
"$table" "${results[@]}" > "$out/matrix.tsv"
printf 'Table: %s/matrix.tsv\n' "$out"
