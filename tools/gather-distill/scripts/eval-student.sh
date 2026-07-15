#!/usr/bin/env bash
# Evaluate one student checkpoint against the fixed 40-job gold set.
#
# Usage:
#   scripts/eval-student.sh CHECKPOINT_OR_GGUF MODEL_LABEL
#
# llama.cpp baseline: build 9580 / b4e3dc613. Keep this pin (or set
# LLAMA_CPP_REVISION) because Qwen3.5 and Gemma 4 text conversion require a
# current converter. This script invokes only the text-model conversion path;
# it never builds or loads a multimodal projector.
#
# Local Mac defaults: -ngl 99 uses Metal, server listens on 127.0.0.1:8090,
# and the harness uses data/eval-*.jsonl plus the local AFT binary. On a CUDA
# host, set LLAMA_CPP_DIR or LLAMA_SERVER/LLAMA_QUANTIZE to that host's build;
# the same -ngl 99 default offloads all layers. A remote GPU can instead serve
# an already-prepared model: set EVAL_REMOTE_ENDPOINT=user@host (and optional
# EVAL_REMOTE_SSH_PORT/EVAL_REMOTE_PORT). The harness then opens ssh -L and
# runs on this machine against the forwarded endpoint.
#
# Important topology: gather spawns GATHER_DISTILL_AFT_BINARY and each eval
# job points at ~/Work/OSS/gather-corpus-eval. Remote harness runs therefore
# need that corpus, data/eval-*.jsonl, and the pinned AFT binary staged there.
# The intended v1 setup is a GPU-host llama-server with this script's harness
# on the Mac through EVAL_REMOTE_ENDPOINT.
#
# Common overrides:
#   EVAL_DATA_DIR               directory containing eval-jobs.jsonl and gold
#   EVAL_OUTPUT_DIR             default: $EVAL_DATA_DIR/students
#   EVAL_CONFIG_JSON            required beside a standalone GGUF to clamp ctx
#   EVAL_CONTEXT_SIZE           request a smaller serving context deliberately
#   EVAL_CHAT_TEMPLATE_KWARGS   JSON passed to llama-server --chat-template-kwargs
#   EVAL_RESET=1                discard this label's resumable run artifacts
#   EVAL_REMOTE_ENDPOINT        ssh destination for a pre-running remote server
#   GATHER_DISTILL_AFT_BINARY   pinned AFT executable used by the harness
#   LLAMA_CPP_DIR               source/build root for converter and binaries

set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "usage: $0 CHECKPOINT_OR_GGUF MODEL_LABEL" >&2
  exit 2
}

die() {
  echo "eval-student: $*" >&2
  exit 1
}

[[ $# -eq 2 ]] || usage
MODEL_INPUT="$1"
LABEL="$2"
[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "MODEL_LABEL may contain only letters, digits, dot, underscore, and dash"

EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ROOT/data}"
EVAL_JOBS="${EVAL_JOBS:-$EVAL_DATA_DIR/eval-jobs.jsonl}"
EVAL_GOLD="${EVAL_GOLD:-$EVAL_DATA_DIR/eval-gold-rows.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$EVAL_DATA_DIR/students}"
EVAL_CORPUS_ROOT="${EVAL_CORPUS_ROOT:-~/Work/OSS/gather-corpus-eval}"
EVAL_TARGET_CONTEXT="${EVAL_TARGET_CONTEXT:-131072}"
EVAL_REQUEST_TIMEOUT="${EVAL_REQUEST_TIMEOUT:-600}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-2}"
EVAL_HOST="${EVAL_HOST:-127.0.0.1}"
EVAL_PORT="${EVAL_PORT:-8090}"
EVAL_GPU_LAYERS="${EVAL_GPU_LAYERS:-99}"
EVAL_CHAT_TEMPLATE_KWARGS="${EVAL_CHAT_TEMPLATE_KWARGS:-}"
LLAMA_CPP_REVISION="${LLAMA_CPP_REVISION:-b4e3dc613}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT/bin/llama.cpp}"
LLAMA_CONVERT="${LLAMA_CONVERT:-$LLAMA_CPP_DIR/convert_hf_to_gguf.py}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_DIR/build/bin/llama-server}"

for file in "$EVAL_JOBS" "$EVAL_GOLD"; do
  [[ -f "$file" ]] || die "missing eval input $file; stage the ignored data/ directory or set EVAL_DATA_DIR"
done
command -v bun >/dev/null || die "bun is required"
command -v python3 >/dev/null || die "python3 is required"
command -v curl >/dev/null || die "curl is required to wait for llama-server"

export GATHER_DISTILL_AFT_BINARY="${GATHER_DISTILL_AFT_BINARY:-$ROOT/bin/aft-dev-7cabfdd0}"
[[ -x "$GATHER_DISTILL_AFT_BINARY" ]] || die "missing executable AFT binary $GATHER_DISTILL_AFT_BINARY; stage it or set GATHER_DISTILL_AFT_BINARY"

mkdir -p "$EVAL_OUTPUT_DIR"
ROWS="$EVAL_OUTPUT_DIR/${LABEL}-rows.jsonl"
LEDGER="$EVAL_OUTPUT_DIR/${LABEL}-ledger.jsonl"
STATUS="$EVAL_OUTPUT_DIR/${LABEL}-status.json"
SCORES="$EVAL_OUTPUT_DIR/${LABEL}-scores.json"
LADDER="$EVAL_OUTPUT_DIR/LADDER.md"
SERVER_LOG="$EVAL_OUTPUT_DIR/${LABEL}-server.log"

if [[ "${EVAL_RESET:-0}" == "1" ]]; then
  rm -f "$ROWS" "$LEDGER" "$STATUS" "$SCORES" "$SERVER_LOG"
fi

CHILD_PID=""
cleanup() {
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

resolve_binary() {
  local configured="$1"
  local fallback="$2"
  if [[ -x "$configured" ]]; then
    printf '%s\n' "$configured"
    return
  fi
  if command -v "$fallback" >/dev/null 2>&1; then
    command -v "$fallback"
    return
  fi
  die "cannot find $fallback; set its explicit environment variable"
}

trained_context_from_config() {
  local config="$1"
  python3 - "$config" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
for key in ("max_position_embeddings", "max_sequence_length", "n_positions", "max_seq_len", "seq_length"):
    value = config.get(key)
    if isinstance(value, int) and value > 0:
        print(value)
        break
else:
    rope = config.get("rope_scaling")
    if isinstance(rope, dict):
        for key in ("original_max_position_embeddings", "original_max_sequence_length"):
            value = rope.get(key)
            if isinstance(value, int) and value > 0:
                print(value)
                break
PY
}

clamp_context() {
  local requested="$1"
  local trained="$2"
  [[ "$requested" =~ ^[1-9][0-9]*$ ]] || die "EVAL_CONTEXT_SIZE/EVAL_TARGET_CONTEXT must be a positive integer"
  if [[ -n "$trained" && "$trained" =~ ^[1-9][0-9]*$ ]] && (( requested > trained )); then
    printf '%s\n' "$trained"
  else
    printf '%s\n' "$requested"
  fi
}

wait_for_server() {
  local base_url="$1"
  local health_url="${base_url%/v1}/health"
  for _ in $(seq 1 120); do
    if curl --fail --silent --show-error --max-time 2 "$health_url" >/dev/null 2>&1 \
      || curl --fail --silent --show-error --max-time 2 "$base_url/models" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  die "llama-server did not become healthy at $base_url; see $SERVER_LOG"
}

SERVER_MODE="${EVAL_SERVER_MODE:-local}"
if [[ -n "${EVAL_REMOTE_ENDPOINT:-}" ]]; then
  SERVER_MODE="tunnel"
fi

case "$SERVER_MODE" in
  local)
    [[ -e "$MODEL_INPUT" ]] || die "model input does not exist: $MODEL_INPUT"
    CONFIG_JSON="${EVAL_CONFIG_JSON:-}"
    MODELS_DIR="$EVAL_OUTPUT_DIR/models"
    mkdir -p "$MODELS_DIR"

    if [[ -d "$MODEL_INPUT" ]]; then
      [[ -n "$CONFIG_JSON" ]] || CONFIG_JSON="$MODEL_INPUT/config.json"
      [[ -f "$CONFIG_JSON" ]] || die "HF checkpoint needs config.json; set EVAL_CONFIG_JSON if it is elsewhere"
      [[ -f "$LLAMA_CONVERT" ]] || die "cannot find convert_hf_to_gguf.py at $LLAMA_CONVERT; set LLAMA_CONVERT or LLAMA_CPP_DIR"
      F16_GGUF="${EVAL_F16_GGUF:-$MODELS_DIR/${LABEL}-F16.gguf}"
      if [[ ! -f "$F16_GGUF" || "${EVAL_RECONVERT:-0}" == "1" ]]; then
        echo "eval-student: converting text checkpoint with llama.cpp $LLAMA_CPP_REVISION"
        python3 "$LLAMA_CONVERT" "$MODEL_INPUT" --outfile "$F16_GGUF" --outtype f16
      fi
      SOURCE_GGUF="$F16_GGUF"
    elif [[ "$MODEL_INPUT" == *.gguf && -f "$MODEL_INPUT" ]]; then
      SOURCE_GGUF="$MODEL_INPUT"
      [[ -n "$CONFIG_JSON" ]] || CONFIG_JSON="$(dirname "$MODEL_INPUT")/config.json"
      [[ -f "$CONFIG_JSON" ]] || die "standalone GGUF needs EVAL_CONFIG_JSON so serving context can be clamped to trained maximum"
    else
      die "model input must be an HF checkpoint directory or a .gguf file"
    fi

    TRAINED_CONTEXT="$(trained_context_from_config "$CONFIG_JSON")"
    SERVED_CONTEXT="$(clamp_context "${EVAL_CONTEXT_SIZE:-$EVAL_TARGET_CONTEXT}" "$TRAINED_CONTEXT")"
    if [[ -n "$TRAINED_CONTEXT" ]]; then
      echo "eval-student: trained context $TRAINED_CONTEXT; serving context $SERVED_CONTEXT"
    else
      echo "eval-student: config has no recognized context key; serving requested context $SERVED_CONTEXT"
    fi

    INPUT_IS_Q8="${EVAL_INPUT_IS_Q8:-}"
    if [[ -z "$INPUT_IS_Q8" ]]; then
      case "$(basename "$SOURCE_GGUF")" in
        *Q8_0*.gguf|*q8_0*.gguf|*Q8-0*.gguf|*q8-0*.gguf) INPUT_IS_Q8=1 ;;
        *) INPUT_IS_Q8=0 ;;
      esac
    fi
    if [[ "$INPUT_IS_Q8" == "1" ]]; then
      SERVED_GGUF="$SOURCE_GGUF"
      echo "eval-student: using existing Q8_0 GGUF $SERVED_GGUF"
    else
      QUANTIZE_BIN="$(resolve_binary "$LLAMA_QUANTIZE" "llama-quantize")"
      SERVED_GGUF="${EVAL_Q8_GGUF:-$MODELS_DIR/${LABEL}-Q8_0.gguf}"
      if [[ ! -f "$SERVED_GGUF" || "${EVAL_REQUANTIZE:-0}" == "1" ]]; then
        QUANTIZE_ARGS=()
        if [[ "${EVAL_ALLOW_REQUANTIZE:-0}" == "1" ]]; then
          QUANTIZE_ARGS+=(--allow-requantize)
        fi
        echo "eval-student: quantizing to Q8_0"
        "$QUANTIZE_BIN" "${QUANTIZE_ARGS[@]}" "$SOURCE_GGUF" "$SERVED_GGUF" Q8_0
      fi
    fi

    SERVER_BIN="$(resolve_binary "$LLAMA_SERVER" "llama-server")"
    SERVER_ARGS=(-m "$SERVED_GGUF" --host "$EVAL_HOST" --port "$EVAL_PORT" -ngl "$EVAL_GPU_LAYERS" --jinja -fa on -c "$SERVED_CONTEXT")
    if [[ -n "$EVAL_CHAT_TEMPLATE_KWARGS" ]]; then
      SERVER_ARGS+=(--chat-template-kwargs "$EVAL_CHAT_TEMPLATE_KWARGS")
    fi
    echo "eval-student: starting $SERVER_BIN (llama.cpp $LLAMA_CPP_REVISION)"
    "$SERVER_BIN" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
    CHILD_PID="$!"
    BASE_URL="${EVAL_BASE_URL:-http://$EVAL_HOST:$EVAL_PORT/v1}"
    wait_for_server "$BASE_URL"
    ;;
  tunnel)
    [[ -n "${EVAL_REMOTE_ENDPOINT:-}" ]] || die "tunnel mode requires EVAL_REMOTE_ENDPOINT=user@host"
    EVAL_TUNNEL_HOST="${EVAL_TUNNEL_HOST:-127.0.0.1}"
    EVAL_TUNNEL_PORT="${EVAL_TUNNEL_PORT:-$EVAL_PORT}"
    EVAL_REMOTE_PORT="${EVAL_REMOTE_PORT:-$EVAL_PORT}"
    SSH_PORT_ARGS=()
    if [[ -n "${EVAL_REMOTE_SSH_PORT:-}" ]]; then
      SSH_PORT_ARGS=(-p "$EVAL_REMOTE_SSH_PORT")
    fi
    echo "eval-student: forwarding $EVAL_TUNNEL_HOST:$EVAL_TUNNEL_PORT to $EVAL_REMOTE_ENDPOINT:$EVAL_REMOTE_PORT"
    ssh "${SSH_PORT_ARGS[@]}" -N -o ExitOnForwardFailure=yes -L "$EVAL_TUNNEL_HOST:$EVAL_TUNNEL_PORT:127.0.0.1:$EVAL_REMOTE_PORT" "$EVAL_REMOTE_ENDPOINT" >"$SERVER_LOG" 2>&1 &
    CHILD_PID="$!"
    BASE_URL="${EVAL_BASE_URL:-http://$EVAL_TUNNEL_HOST:$EVAL_TUNNEL_PORT/v1}"
    SERVED_CONTEXT="${EVAL_CONTEXT_SIZE:-remote-unknown}"
    wait_for_server "$BASE_URL"
    ;;
  *)
    die "EVAL_SERVER_MODE must be local or tunnel"
    ;;
esac

cd "$ROOT"
echo "eval-student: gathering 40 fixed eval jobs through $BASE_URL"
bun run src/cli.ts gather \
  --backend openai \
  --base-url "$BASE_URL" \
  --model "${EVAL_SERVED_MODEL:-$LABEL}" \
  --request-timeout "$EVAL_REQUEST_TIMEOUT" \
  --jobs "$EVAL_JOBS" \
  --concurrency "$EVAL_CONCURRENCY" \
  --inline-validate \
  --rows "$ROWS" \
  --ledger "$LEDGER" \
  --status "$STATUS"

echo "eval-student: scoring against $EVAL_GOLD"
bun run src/cli.ts score \
  --candidate "$ROWS" \
  --gold "$EVAL_GOLD" \
  --output "$SCORES" \
  --corpus-root "$EVAL_CORPUS_ROOT"

python3 - "$SCORES" "$LEDGER" "$LADDER" "$LABEL" "$SERVED_CONTEXT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

scores_path = Path(sys.argv[1])
ledger_path = Path(sys.argv[2])
ladder_path = Path(sys.argv[3])
label = sys.argv[4]
served_context = sys.argv[5]
report = json.loads(scores_path.read_text(encoding="utf-8"))
jobs = [job for job in report.get("jobs", []) if isinstance(job, dict)]


def mean(values):
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


natural = [job for job in jobs if job.get("budget_outcome") == "natural"]
contract_valid = mean(1 if job.get("contract_valid") else 0 for job in jobs)
api_error = mean(1 if job.get("budget_outcome") == "api_error" else 0 for job in jobs)
tool_calls = mean(job.get("candidate_tool_calls") for job in jobs)
thinking_tokens = mean(job.get("thinking_tokens") for job in jobs)
natural_file_f1 = mean(job.get("file_f1") for job in natural)
natural_line_jaccard = mean(job.get("line_overlap") for job in natural)
budgets = Counter(str(job.get("budget_outcome", "unknown")) for job in jobs)

ledger_durations = []
if ledger_path.exists():
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("duration_ms"), (int, float)):
            ledger_durations.append(row["duration_ms"] / 1000)
wall_seconds = mean(ledger_durations)

header = """# Student SFT ladder\n\n| model | natural file F1 | natural line Jaccard | contract-valid rate | API-error rate | avg tool calls | thinking tokens/traj | natural jobs | budget outcomes | served context | wall time/traj |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |\n"""
if not ladder_path.exists() or not ladder_path.read_text(encoding="utf-8").strip():
    ladder_path.parent.mkdir(parents=True, exist_ok=True)
    ladder_path.write_text(header, encoding="utf-8")

budget_text = f"N {budgets['natural']}/F {budgets['budget_finalize']}/A {budgets['api_error']}/I {budgets['invalid_final']}"
row = " | ".join(
    [
        label,
        number(natural_file_f1),
        number(natural_line_jaccard),
        percent(contract_valid),
        percent(api_error),
        number(tool_calls, 2),
        number(thinking_tokens, 0),
        f"{len(natural)}/{len(jobs)}",
        budget_text,
        served_context,
        "n/a" if wall_seconds is None else f"{wall_seconds:.1f}s",
    ]
)
with ladder_path.open("a", encoding="utf-8") as handle:
    handle.write(f"| {row} |\n")

print(
    json.dumps(
        {
            "lane": "eval-student",
            "label": label,
            "scores": str(scores_path),
            "ladder": str(ladder_path),
            "natural_file_f1": natural_file_f1,
            "natural_line_jaccard": natural_line_jaccard,
            "contract_valid_rate": contract_valid,
            "api_error_rate": api_error,
            "avg_tool_calls": tool_calls,
            "served_context": served_context,
        }
    )
)
PY
