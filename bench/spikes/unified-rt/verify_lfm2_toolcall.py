#!/usr/bin/env python3
"""Verify LFM2 bracket tool calls through the owned runtime and llama.cpp.

The harness deliberately renders prompts with the model's Hugging Face chat
 template before either engine sees them.  It parses the generated bracket body
with Python's AST instead of accepting a marker-only or regex-only match.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

TOOL_LIST_START = "<|tool_list_start|>"
TOOL_LIST_END = "<|tool_list_end|>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Create a reminder at a local time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "time": {"type": "string"},
                    "timezone": {"type": "string"},
                },
                "required": ["title", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]

PROMPTS: list[tuple[str, str, str]] = [
    (
        "weather-istanbul",
        "get_weather",
        "What's the weather in Istanbul? Use the get_weather tool and then stop.",
    ),
    (
        "weather-ankara",
        "get_weather",
        "What's the current weather in Ankara in Fahrenheit? Call get_weather; do not answer in prose.",
    ),
    (
        "reminder-call-mom",
        "create_reminder",
        "Set a reminder titled 'Call mom' for 2026-07-21T09:00:00 in Europe/Istanbul. Use create_reminder.",
    ),
    (
        "currency-usd-try",
        "convert_currency",
        "Convert 100 USD to TRY. Use convert_currency and return the tool call only.",
    ),
    (
        "reminder-groceries",
        "create_reminder",
        "Create a reminder titled 'Buy groceries' for 2026-07-21T18:30:00. Call create_reminder only.",
    ),
]

SYSTEM_PROMPT = (
    "You are a precise tool-using assistant. When a tool is needed, emit only "
    "the LFM2 bracket-notation tool call and no prose."
)


@dataclass
class ParsedCall:
    name: str
    keywords: dict[str, Any]
    positional_count: int
    marker_wrapped: bool

    @property
    def structure(self) -> tuple[str, int, tuple[str, ...]]:
        return (self.name, self.positional_count, tuple(sorted(self.keywords)))


@dataclass
class Check:
    valid: bool
    parsed: ParsedCall | None
    reason: str
    transcript: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned LFM2 HF snapshot")
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("target/release/spike-unified-rt"),
        help="Built spike-unified-rt binary",
    )
    parser.add_argument(
        "--llama-cli",
        type=Path,
        default=Path(shutil.which("llama-cli") or "llama-cli"),
    )
    parser.add_argument("--gguf", type=Path, required=True, help="Official llama.cpp GGUF")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("target/lfm2-toolcall-verification"),
        help="Directory for rendered prompts and engine JSON/transcripts",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-owned-valid", type=int, default=3)
    parser.add_argument("--min-agreements", type=int, default=3)
    return parser.parse_args()


def literal_value(node: ast.AST) -> Any:
    """Evaluate only literal Python values accepted as tool arguments."""

    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError) as error:
        raise ValueError(f"argument is not a literal: {ast.dump(node)}") from error


def call_from_node(node: ast.AST, marker_wrapped: bool) -> ParsedCall:
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        raise ValueError("tool-call body must be a non-empty list")
    if len(node.elts) != 1:
        raise ValueError("verification prompts must produce exactly one tool call")
    call = node.elts[0]
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError("list element is not a simple Python call")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("starred keyword arguments are not supported")
    keywords = {keyword.arg: literal_value(keyword.value) for keyword in call.keywords}
    return ParsedCall(
        name=call.func.id,
        keywords=keywords,
        positional_count=len(call.args),
        marker_wrapped=marker_wrapped,
    )


def balanced_list_fragments(text: str) -> list[str]:
    """Return balanced bracket expressions without treating brackets as regex."""

    fragments: list[str] = []
    for start, character in enumerate(text):
        if character != "[":
            continue
        depth = 0
        quote: str | None = None
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if quote is not None:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == quote:
                    quote = None
                continue
            if current in ("'", '"'):
                quote = current
            elif current == "[":
                depth += 1
            elif current == "]":
                depth -= 1
                if depth == 0:
                    fragments.append(text[start : end + 1])
                    break
    return fragments


def parse_transcript(text: str, require_markers: bool) -> ParsedCall:
    marker_wrapped = TOOL_CALL_START in text or TOOL_CALL_END in text
    if require_markers and not marker_wrapped:
        raise ValueError("missing tool-call markers")
    if marker_wrapped:
        if text.count(TOOL_CALL_START) != 1 or text.count(TOOL_CALL_END) != 1:
            raise ValueError("tool-call markers are not paired exactly once")
        start = text.index(TOOL_CALL_START) + len(TOOL_CALL_START)
        end = text.index(TOOL_CALL_END)
        if end < start:
            raise ValueError("tool-call end marker precedes start marker")
        body = text[start:end].strip()
        try:
            expression = ast.parse(body, mode="eval").body
        except SyntaxError as error:
            raise ValueError(f"tool-call body is not Python syntax: {body!r}") from error
        return call_from_node(expression, marker_wrapped=True)

    for fragment in balanced_list_fragments(text):
        try:
            expression = ast.parse(fragment, mode="eval").body
            if isinstance(expression, (ast.List, ast.Tuple)) and expression.elts and all(
                isinstance(element, ast.Call) for element in expression.elts
            ):
                return call_from_node(expression, marker_wrapped=False)
        except (SyntaxError, ValueError):
            continue
    raise ValueError("no parseable Python call list found")


def tool_contracts() -> dict[str, tuple[set[str], set[str]]]:
    contracts: dict[str, tuple[set[str], set[str]]] = {}
    for tool in TOOLS:
        function = tool["function"]
        parameters = function["parameters"]
        contracts[function["name"]] = (
            set(parameters["required"]),
            set(parameters["properties"]),
        )
    return contracts


def validate_call(parsed: ParsedCall, expected_name: str) -> str | None:
    contracts = tool_contracts()
    if parsed.name != expected_name:
        return f"expected {expected_name}, got {parsed.name}"
    if parsed.positional_count:
        return "positional arguments are not accepted by the documented tool schemas"
    if parsed.name not in contracts:
        return f"unknown tool {parsed.name}"
    required, properties = contracts[parsed.name]
    missing = required - parsed.keywords.keys()
    unknown = parsed.keywords.keys() - properties
    if missing:
        return f"missing required argument(s): {', '.join(sorted(missing))}"
    if unknown:
        return f"unknown argument(s): {', '.join(sorted(unknown))}"
    for name, value in parsed.keywords.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return f"argument {name} is not a scalar literal"
    if parsed.name == "get_weather" and "units" in parsed.keywords:
        if parsed.keywords["units"] not in {"celsius", "fahrenheit"}:
            return "weather units are outside the declared enum"
    return None


def render_prompts(tokenizer: Any, artifact_dir: Path) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    prompt_path = artifact_dir / "rendered-prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8") as stream:
        for prompt_id, expected_name, user in PROMPTS:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                tools=TOOLS,
                add_generation_prompt=True,
                tokenize=True,
            )
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=TOOLS,
                add_generation_prompt=True,
                tokenize=False,
            )
            runtime_encoded = tokenizer(prompt, add_special_tokens=True)
            row = {
                "id": prompt_id,
                "expected_tool": expected_name,
                "user": user,
                "messages": messages,
                "tools": TOOLS,
                "prompt": prompt,
                "template_input_ids": [int(token) for token in encoded["input_ids"]],
                # The owned loader calls tokenizer.encode(text, true), so keep
                # the exact runtime input ids as a separate fixture field.
                "input_ids": [int(token) for token in runtime_encoded["input_ids"]],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rendered.append(row)
    return rendered


def run_owned(args: argparse.Namespace, tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    artifact_dir: Path = args.artifact_dir
    prompt_path = artifact_dir / "owned-prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({"id": row["id"], "prompt": row["prompt"]}) + "\n")
    output_path = artifact_dir / "owned.json"
    command = [
        str(args.runtime),
        "--model",
        str(args.model),
        "--tokenizer",
        str(args.model / "tokenizer.json"),
        "--generate-prompts",
        str(prompt_path),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--decode-cache-bucket",
        "512",
        "--decode-top-k",
        "1",
        "--device",
        "cpu",
        "--dtype",
        "f32",
        "--out",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    (artifact_dir / "owned.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (artifact_dir / "owned.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"owned runtime failed with {completed.returncode}: {completed.stderr[-2000:]}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_constraint_probe(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    """Run JSON masking on one tool prompt to expose span-format interaction."""

    prompt_path = args.artifact_dir / "constraint-probe-prompt.jsonl"
    prompt_path.write_text(
        json.dumps({"id": row["id"], "prompt": row["prompt"]}) + "\n", encoding="utf-8"
    )
    output_path = args.artifact_dir / "constraint-probe.json"
    command = [
        str(args.runtime),
        "--model",
        str(args.model),
        "--tokenizer",
        str(args.model / "tokenizer.json"),
        "--generate-prompts",
        str(prompt_path),
        "--decode-json",
        "--max-new-tokens",
        "64",
        "--decode-cache-bucket",
        "512",
        "--decode-top-k",
        "1",
        "--device",
        "cpu",
        "--dtype",
        "f32",
        "--out",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    (args.artifact_dir / "constraint-probe.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (args.artifact_dir / "constraint-probe.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(
            f"JSON constraint probe failed with {completed.returncode}: {completed.stderr[-2000:]}"
        )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if result.get("constraint") != "json" or result.get("constraint_valid_prompts") != 1:
        raise RuntimeError(f"JSON constraint probe did not report one valid JSON result: {result}")
    text = result["results"][0].get("text")
    if not isinstance(text, str):
        raise RuntimeError("JSON constraint probe did not return decoded text")
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"JSON constraint probe returned invalid JSON: {text!r}") from error
    result["probe_text"] = text
    result["tool_markers_in_probe_text"] = TOOL_CALL_START in text or TOOL_CALL_END in text
    return result


def run_llama(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for row in rows:
        prompt_path = args.artifact_dir / f"llama-{row['id']}.txt"
        prompt_path.write_text(row["prompt"], encoding="utf-8")
        command = [
            str(args.llama_cli),
            "-m",
            str(args.gguf),
            "-f",
            str(prompt_path),
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
            "--temp",
            "0",
            "--top-k",
            "1",
            "--top-p",
            "1",
            "--n-predict",
            str(args.max_new_tokens),
            "--seed",
            "42",
            "--log-disable",
            "--no-perf",
            "--no-warmup",
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=300)
        transcript = completed.stdout + completed.stderr
        (args.artifact_dir / f"llama-{row['id']}.transcript").write_text(
            transcript, encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(
                f"llama-cli failed for {row['id']} with {completed.returncode}: {transcript[-2000:]}"
            )
        transcripts[row["id"]] = transcript
    return transcripts


def check_owned(
    tokenizer: Any, owned: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Check]:
    row_by_id = {row["id"]: row for row in rows}
    checks: dict[str, Check] = {}
    for result in owned["results"]:
        row = row_by_id[result["id"]]
        transcript = tokenizer.decode(
            result["tokens"], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        try:
            parsed = parse_transcript(transcript, require_markers=True)
            reason = validate_call(parsed, row["expected_tool"])
            checks[result["id"]] = Check(reason is None, parsed, reason or "valid", transcript)
        except ValueError as error:
            checks[result["id"]] = Check(False, None, str(error), transcript)
        if result["prompt_tokens"] != len(row["input_ids"]):
            raise RuntimeError(
                f"{result['id']} prompt token mismatch: runtime {result['prompt_tokens']} "
                f"vs fixture {len(row['input_ids'])}"
            )
    if set(checks) != set(row_by_id):
        raise RuntimeError("owned runtime did not return exactly one row per rendered prompt")
    return checks


def check_llama(transcripts: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Check]:
    checks: dict[str, Check] = {}
    for row in rows:
        transcript = transcripts[row["id"]]
        try:
            parsed = parse_transcript(transcript, require_markers=False)
            reason = validate_call(parsed, row["expected_tool"])
            checks[row["id"]] = Check(reason is None, parsed, reason or "valid", transcript)
        except ValueError as error:
            checks[row["id"]] = Check(False, None, str(error), transcript)
    return checks


def main() -> int:
    args = parse_args()
    if len(PROMPTS) < 5:
        raise RuntimeError("verification gate requires at least five prompts")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    for path, label in ((args.model, "model"), (args.runtime, "runtime"), (args.gguf, "GGUF")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = render_prompts(tokenizer, args.artifact_dir)
    marker_ids = {
        marker: int(tokenizer.convert_tokens_to_ids(marker))
        for marker in (
            TOOL_LIST_START,
            TOOL_LIST_END,
            TOOL_CALL_START,
            TOOL_CALL_END,
            "<|tool_response_start|>",
            "<|tool_response_end|>",
        )
    }
    (args.artifact_dir / "template-metadata.json").write_text(
        json.dumps(
            {
                "transformers": __import__("transformers").__version__,
                "model": str(args.model),
                "special_token_ids": marker_ids,
                "prompt_count": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    owned = run_owned(args, tokenizer, rows)
    owned_checks = check_owned(tokenizer, owned, rows)
    constraint_probe = run_constraint_probe(args, rows[0])
    llama_transcripts = run_llama(args, rows)
    llama_checks = check_llama(llama_transcripts, rows)

    owned_valid = sum(check.valid for check in owned_checks.values())
    llama_valid = sum(check.valid for check in llama_checks.values())
    agreements = sum(
        owned_checks[row["id"]].valid
        and llama_checks[row["id"]].valid
        and owned_checks[row["id"]].parsed is not None
        and llama_checks[row["id"]].parsed is not None
        and owned_checks[row["id"]].parsed.structure[:2]
        == llama_checks[row["id"]].parsed.structure[:2]
        for row in rows
    )
    summary = {
        "prompt_count": len(rows),
        "owned_valid": owned_valid,
        "llama_valid": llama_valid,
        "structural_agreements": agreements,
        "constraint_probe": {
            "constraint": constraint_probe["constraint"],
            "constraint_valid_prompts": constraint_probe["constraint_valid_prompts"],
            "text": constraint_probe["probe_text"],
            "tool_markers_in_text": constraint_probe["tool_markers_in_probe_text"],
        },
        "owned": {
            prompt_id: {
                "valid": check.valid,
                "reason": check.reason,
                "transcript": check.transcript,
                "structure": check.parsed.structure if check.parsed else None,
            }
            for prompt_id, check in owned_checks.items()
        },
        "llama": {
            prompt_id: {
                "valid": check.valid,
                "reason": check.reason,
                "transcript": check.transcript,
                "structure": check.parsed.structure if check.parsed else None,
            }
            for prompt_id, check in llama_checks.items()
        },
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("prompt_count", "owned_valid", "llama_valid", "structural_agreements")}))
    for prompt_id in (row["id"] for row in rows):
        owned_check = owned_checks[prompt_id]
        llama_check = llama_checks[prompt_id]
        print(
            f"{prompt_id}: owned={'PASS' if owned_check.valid else 'FAIL'} "
            f"({owned_check.reason}); llama={'PASS' if llama_check.valid else 'FAIL'} "
            f"({llama_check.reason})"
        )
    if owned_valid < args.min_owned_valid:
        raise RuntimeError(f"owned valid calls {owned_valid} < required {args.min_owned_valid}")
    if llama_valid < args.min_agreements or agreements < args.min_agreements:
        raise RuntimeError(
            f"llama valid/agreement counts {llama_valid}/{agreements} "
            f"< required {args.min_agreements}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"lfm2 tool-call verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
