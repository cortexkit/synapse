#!/usr/bin/env python3
"""Validate the immutable agentic battery before recording an acceptance run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    """Raised when the fixture no longer matches its recorded identity."""


PROMPT_COUNTS = {
    "tool-call-transcript": 5,
    "code-generation": 5,
    "constrained-json": 5,
    "long-context-prose-continuation": 5,
}

EXPECTED_INPUT_TOKENS = {
    "tool-call-transcript-01": 1024,
    "tool-call-transcript-02": 1536,
    "tool-call-transcript-03": 2048,
    "tool-call-transcript-04": 3072,
    "tool-call-transcript-05": 4096,
    "code-generation-01": 1536,
    "code-generation-02": 2048,
    "code-generation-03": 3072,
    "code-generation-04": 4096,
    "code-generation-05": 6144,
    "constrained-json-01": 2048,
    "constrained-json-02": 3072,
    "constrained-json-03": 4096,
    "constrained-json-04": 6144,
    "constrained-json-05": 8192,
    "long-context-prose-continuation-01": 4096,
    "long-context-prose-continuation-02": 6144,
    "long-context-prose-continuation-03": 8192,
    "long-context-prose-continuation-04": 12288,
    "long-context-prose-continuation-05": 16384,
}

PINNED_FIELDS = {
    "schema_version": "agentic-battery-manifest-v1",
    "battery_revision": "agentic-battery-v1",
    "write_once": True,
    "prompt_count": 20,
    "prompt_digest_algorithm": "sha256",
    "input_token_range": {"minimum": 1024, "maximum": 16384},
    "eligibility_status": "pending_oracle_run",
    "tokenizer": {
        "id": "Qwen/Qwen3.8-27B",
        "revision": "b6c73d81f0d4fc313f6e03d7cae5b35d1a52af84",
        "sha256": "5b3e0a821a4a781b0f86a1db82948fd82464941ae4134d5bc2c4c6658a51c51f",
        "token_counting": "apply_chat_template(add_generation_prompt=true) then encode",
    },
    "chat_template": {
        "id": "qwen3.8-instruct-tool-use-v1",
        "sha256": "53a1e6ee6a0bd8f6a7fb2784e5e292482d8d5a742f2fe41e524b39295c1f63ec",
        "add_generation_prompt": True,
    },
    "generation": {
        "mode": "greedy",
        "do_sample": False,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "min_p": 0,
        "repetition_penalty": 1,
        "seed": 0,
        "max_new_tokens": 256,
        "unsupported_non_greedy_error": "owned_decode_sampling_unsupported",
    },
    "artifact": {
        "id": "qwen3.8-27b-q4-k-m-native-mtp-v1",
        "model": "Qwen3.8-27B",
        "format": "GGUF",
        "quantization": "Q4_K_M",
        "pinned": False,
        "source_sha256": None,
        "catalog_fingerprint": None,
        "binds_at": "first-ingest",
    },
    "band_gate": {
        "id": "agentic-battery-exact-token-parity-v1",
        "mode": "exact",
        "fingerprint": "fa8ed7d4a424e88d9492f4d0d94aeec9340ec9fd6088fddf039f072521eb415a",
        "near_tie_policy": "disabled",
    },
    "harness": {
        "revision": "agentic-battery-harness-v1",
        "entrypoint": "harness.py",
        "result_schema": "agentic-battery-acceptance-result-v1",
    },
    "llama_cpp_oracle": {
        "repository": "ggml-org/llama.cpp",
        "revision": "b9580",
        "backend": "metal",
        "build_flags": [
            "-DGGML_METAL=ON",
            "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DGGML_NATIVE=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
    },
}

REQUIRED_MANIFEST_FIELDS = [
    "schema_version",
    "battery_revision",
    "write_once",
    "prompt_count",
    "prompt_digest_algorithm",
    "input_token_range",
    "eligibility_status",
    "prompts",
    "tokenizer",
    "chat_template",
    "generation",
    "artifact",
    "band_gate",
    "harness",
    "harness_sha256",
    "llama_cpp_oracle",
    "manifest_content_sha256",
]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """Serialize the manifest without its self-referential content digest."""
    canonical = copy.deepcopy(manifest)
    canonical.pop("manifest_content_sha256", None)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def read_pinned_digest(path: Path, expected_name: str) -> str:
    try:
        fields = path.read_text(encoding="utf-8").strip().split()
    except OSError as error:
        raise ManifestError(f"cannot read digest file {path}: {error}") from error
    if len(fields) != 2 or fields[1] != expected_name:
        raise ManifestError(f"{path} must contain one digest for {expected_name}")
    digest = fields[0]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ManifestError(f"{path} does not contain a lowercase SHA-256 digest")
    return digest


def validate_pin(manifest: dict[str, Any], field: str, expected: Any) -> None:
    actual = manifest.get(field)
    if actual is None:
        raise ManifestError(f"required manifest field is absent: {field}")
    if actual != expected:
        raise ManifestError(f"required manifest field does not match its pin: {field}")


def validate_prompts(manifest: dict[str, Any], manifest_path: Path) -> None:
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 20:
        raise ManifestError("prompts must be a list containing exactly 20 entries")

    seen_ids: set[str] = set()
    observed_counts = {category: 0 for category in PROMPT_COUNTS}
    observed_tokens: list[int] = []
    for entry in prompts:
        if not isinstance(entry, dict):
            raise ManifestError("every prompt entry must be an object")
        for field in (
            "id",
            "category",
            "path",
            "input_tokens",
            "sha256",
            "speed_gate_eligible",
        ):
            if field not in entry:
                raise ManifestError(f"prompt entry is missing required field: {field}")

        prompt_id = entry["id"]
        category = entry["category"]
        path_text = entry["path"]
        input_tokens = entry["input_tokens"]
        digest = entry["sha256"]
        speed_gate_eligible = entry["speed_gate_eligible"]
        if not isinstance(prompt_id, str) or prompt_id in seen_ids:
            raise ManifestError(f"prompt id must be unique: {prompt_id!r}")
        if category not in PROMPT_COUNTS:
            raise ManifestError(f"prompt {prompt_id} has an unknown category: {category!r}")
        if not isinstance(path_text, str) or not path_text.startswith("prompts/"):
            raise ManifestError(f"prompt {prompt_id} must use a prompts/ relative path")
        prompt_path = manifest_path.parent / path_text
        if prompt_path.resolve().parent != (manifest_path.parent / "prompts").resolve():
            raise ManifestError(f"prompt {prompt_id} path escapes the prompts directory")
        if not isinstance(input_tokens, int) or input_tokens != EXPECTED_INPUT_TOKENS.get(prompt_id):
            raise ManifestError(f"prompt {prompt_id} has an unpinned input token count")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ManifestError(f"prompt {prompt_id} has no SHA-256 digest")
        if speed_gate_eligible is not None and not isinstance(speed_gate_eligible, bool):
            raise ManifestError(f"prompt {prompt_id} has an invalid speed-gate eligibility value")

        try:
            prompt_bytes = prompt_path.read_bytes()
        except OSError as error:
            raise ManifestError(f"prompt {prompt_id} cannot be read: {error}") from error
        if sha256_hex(prompt_bytes) != digest:
            raise ManifestError(f"prompt {prompt_id} digest does not match its fixture bytes")
        prompt = read_json(prompt_path)
        if (
            prompt.get("id") != prompt_id
            or prompt.get("category") != category
            or prompt.get("input_tokens") != input_tokens
            or prompt.get("speed_gate_eligible") != speed_gate_eligible
            or not isinstance(prompt.get("messages"), list)
        ):
            raise ManifestError(f"prompt {prompt_id} does not match its manifest entry")

        seen_ids.add(prompt_id)
        observed_counts[category] += 1
        observed_tokens.append(input_tokens)

    if observed_counts != PROMPT_COUNTS:
        raise ManifestError(f"prompt family composition is not pinned: {observed_counts}")
    if min(observed_tokens) != 1024 or max(observed_tokens) != 16384:
        raise ManifestError("prompt input token range must span 1k through 16k")


def is_speed_gate_acceptance(acceptance: dict[str, Any]) -> bool:
    """Recognize the stable speed-gate labels accepted by the harness input."""
    return acceptance.get("gate") in {"speed-gate", "speed_gate"} or acceptance.get(
        "acceptance_kind"
    ) in {"speed-gate", "speed_gate"}


def validate_speed_gate_eligibility(manifest: dict[str, Any]) -> None:
    """Require an oracle-backed E3 selection before a speed result is recorded."""
    eligible = [entry.get("speed_gate_eligible") for entry in manifest["prompts"]]
    if not all(isinstance(value, bool) for value in eligible):
        raise ManifestError("speed-gate acceptance requires boolean eligibility for every prompt")
    if sum(eligible) < 16:
        raise ManifestError("speed-gate acceptance requires at least 16 eligible prompts")


def validate_acceptance_recording(manifest: dict[str, Any], acceptance: dict[str, Any]) -> None:
    """Fail closed until ingest and the oracle run establish certifying inputs."""
    if is_speed_gate_acceptance(acceptance):
        validate_speed_gate_eligibility(manifest)
    if manifest["artifact"]["pinned"] is not True:
        raise ManifestError("acceptance recording is refused while the artifact is not pinned")


def validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], str]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = read_json(manifest_path)

    required_fields = manifest.get("required_manifest_fields")
    if required_fields != REQUIRED_MANIFEST_FIELDS:
        raise ManifestError("required_manifest_fields is absent or does not match the harness contract")
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            raise ManifestError(f"required manifest field is absent: {field}")
    for field, expected in PINNED_FIELDS.items():
        validate_pin(manifest, field, expected)

    content_digest = manifest.get("manifest_content_sha256")
    if content_digest != sha256_hex(canonical_manifest_bytes(manifest)):
        raise ManifestError("manifest_content_sha256 does not match the canonical manifest")
    raw_digest = sha256_hex(manifest_bytes)
    if read_pinned_digest(manifest_path.with_suffix(".sha256"), manifest_path.name) != raw_digest:
        raise ManifestError("manifest.sha256 does not match manifest.json")

    harness = manifest["harness"]
    harness_path = manifest_path.parent / harness["entrypoint"]
    if harness_path.resolve() != Path(__file__).resolve():
        raise ManifestError("manifest harness entrypoint does not identify this harness")
    harness_digest = sha256_hex(harness_path.read_bytes())
    if manifest.get("harness_sha256") != harness_digest:
        raise ManifestError("harness SHA-256 does not match the manifest")
    if read_pinned_digest(harness_path.with_suffix(".sha256"), harness_path.name) != harness_digest:
        raise ManifestError("harness.sha256 does not match harness.py")

    validate_prompts(manifest, manifest_path)
    return manifest, raw_digest


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--result", type=Path)
    parser.add_argument(
        "--acceptance-input",
        type=Path,
        help="optional JSON object from the caller to attach after fixture validation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate fixture integrity without recording an acceptance result",
    )
    args = parser.parse_args()
    if not args.validate_only and args.result is None:
        parser.error("--result is required unless --validate-only is used")
    if args.validate_only and args.result is not None:
        parser.error("--validate-only cannot be combined with --result")

    try:
        manifest, manifest_digest = validate_manifest(args.manifest)
        if args.validate_only:
            return 0
        acceptance: dict[str, Any] = {}
        if args.acceptance_input is not None:
            acceptance = read_json(args.acceptance_input)
        validate_acceptance_recording(manifest, acceptance)
        write_result(
            args.result,
            {
                "schema_version": manifest["harness"]["result_schema"],
                "status": "accepted",
                "battery_revision": manifest["battery_revision"],
                "manifest_sha256": manifest_digest,
                "manifest_content_sha256": manifest["manifest_content_sha256"],
                "harness_revision": manifest["harness"]["revision"],
                "prompt_count": manifest["prompt_count"],
                "acceptance": acceptance,
            },
        )
    except ManifestError as error:
        parser.error(f"manifest rejected: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
