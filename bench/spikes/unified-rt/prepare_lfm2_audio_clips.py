#!/usr/bin/env python3
"""Create a deterministic 20-clip macOS `say` ASR correctness corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A small silver airplane crossed the bright morning sky.",
    "Please leave the package beside the wooden front door.",
    "We counted seven red apples and three green pears.",
    "The library closes at six o'clock on Friday evening.",
    "Fresh coffee was waiting in the kitchen after sunrise.",
    "Her new bicycle has a bell and a sturdy black basket.",
    "Turn left after the bridge and continue for two miles.",
    "The children built a tall castle from damp beach sand.",
    "Our train arrives at the central station before noon.",
    "A gentle breeze moved through the open bedroom window.",
    "He wrote the meeting notes in a blue paper notebook.",
    "The weather forecast predicts light rain tomorrow morning.",
    "Four musicians practiced the same melody in the hall.",
    "Keep the yellow ticket until the end of your journey.",
    "The old clock stopped exactly at twenty minutes past nine.",
    "She planted rosemary, mint, and basil in the garden.",
    "A narrow path follows the river around the quiet village.",
    "Remember to charge the camera before the weekend trip.",
    "Warm sunlight filled the room as the curtains opened.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--rate", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lfm2-audio-say-") as temporary:
        temporary_path = Path(temporary)
        rows = []
        for index, sentence in enumerate(SENTENCES, 1):
            clip_id = f"say-{index:02d}"
            aiff = temporary_path / f"{clip_id}.aiff"
            wav = args.out_dir / f"{clip_id}.wav"
            subprocess.run(
                [
                    "say",
                    "--voice",
                    args.voice,
                    "--rate",
                    str(args.rate),
                    "--output-file",
                    str(aiff),
                    sentence,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16@16000",
                    "-c",
                    "1",
                    str(aiff),
                    str(wav),
                ],
                check=True,
            )
            rows.append(
                {
                    "id": clip_id,
                    "path": str(Path(args.out_dir.name) / wav.name),
                    "source_text": sentence,
                }
            )
    with args.manifest.open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    print(
        f"wrote {len(rows)} 16 kHz mono WAV clips to {args.out_dir} "
        f"and manifest {args.manifest}"
    )


if __name__ == "__main__":
    main()
