#!/usr/bin/env python3
"""Synthesize 16 kHz mono bootstrap ASR audio with macOS `say` and `afconvert`."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--voices", default="Samantha,Daniel")
    parser.add_argument("--rate", type=int, default=180)
    parser.add_argument("--limit", type=int, help="Development-only cap on source utterances.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def voice_slug(voice: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in voice).strip("-")


def main() -> None:
    args = parse_args()
    for tool in ("say", "afconvert"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is required; this bootstrap synthesizer runs on macOS")

    rows = load_jsonl(args.utterances)
    if args.limit is not None:
        rows = rows[: args.limit]
    voices = [voice.strip() for voice in args.voices.split(",") if voice.strip()]
    if not voices:
        raise SystemExit("--voices must name at least one macOS say voice")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="stt-bias-say-") as temporary:
        temporary_path = Path(temporary)
        for row in rows:
            for voice in voices:
                slug = voice_slug(voice)
                output_dir = args.out_dir / slug
                output_dir.mkdir(parents=True, exist_ok=True)
                aiff = temporary_path / f"{row['id']}-{slug}.aiff"
                wav = output_dir / f"{row['id']}.wav"
                subprocess.run(
                    [
                        "say",
                        "--voice",
                        voice,
                        "--rate",
                        str(args.rate),
                        "--output-file",
                        str(aiff),
                        row["source_text"],
                    ],
                    check=True,
                )
                subprocess.run(
                    ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
                    check=True,
                )
                manifest_rows.append(
                    {
                        **row,
                        "id": f"{row['id']}-{slug}",
                        "path": str(wav.relative_to(args.manifest.parent)),
                        "voice": voice,
                        "sample_rate_hz": 16000,
                        "channels": 1,
                    }
                )

    with args.manifest.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(manifest_rows)} 16 kHz mono WAV clips to {args.out_dir}")


if __name__ == "__main__":
    main()
