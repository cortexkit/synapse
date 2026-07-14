#!/usr/bin/env python3
"""Pinned liquid-audio oracle for LFM2-Audio ASR parity gates.

The output JSONL is accepted directly by spike-unified-rt --asr-reference. It
contains frame-major normalized mel values, projected audio embeddings, greedy
text token IDs (including the terminal token), and decoded text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

LIQUID_AUDIO_COMMIT = "84cdb243859aaa53db660bc3f4718b54133336bd"
CHECKPOINT_REVISION = "c798aad30dc3cd72e72970beab51326b8443bd94"
WEIGHT_SHA256 = "d0cae5b6a1cbc308472535d6fa310fe446bb9ea601934a14db2366040a9fa129"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--versions-out", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--liquid-audio-source",
        type=Path,
        help="Pinned liquid-audio checkout; verified when supplied.",
    )
    parser.add_argument(
        "--skip-weight-hash",
        action="store_true",
        help="Skip the 2.9 GB safetensors SHA-256 check.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError("input JSONL is empty")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pins(args: argparse.Namespace) -> None:
    if args.model.name != CHECKPOINT_REVISION:
        raise RuntimeError(
            f"model directory must be the pinned snapshot {CHECKPOINT_REVISION}, got {args.model}"
        )
    if not args.skip_weight_hash:
        actual = sha256(args.model / "model.safetensors")
        if actual != WEIGHT_SHA256:
            raise RuntimeError(f"weight SHA-256 {actual} != pinned {WEIGHT_SHA256}")
    if args.liquid_audio_source is not None:
        actual = subprocess.check_output(
            ["git", "-C", str(args.liquid_audio_source), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if actual != LIQUID_AUDIO_COMMIT:
            raise RuntimeError(
                f"liquid-audio checkout {actual} != pinned {LIQUID_AUDIO_COMMIT}"
            )


def versions() -> dict[str, str]:
    packages = [
        "liquid-audio",
        "torch",
        "torchaudio",
        "transformers",
        "accelerate",
        "librosa",
        "numpy",
    ]
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "liquid_audio_commit": LIQUID_AUDIO_COMMIT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "weight_sha256": WEIGHT_SHA256,
    }
    for package in packages:
        result[package] = importlib.metadata.version(package)
    return result


def main() -> None:
    args = parse_args()
    verify_pins(args)

    import soundfile
    import torch
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

    torch.set_grad_enabled(False)
    processor = LFM2AudioProcessor.from_pretrained(args.model, device="cpu").eval()
    model = LFM2AudioModel.from_pretrained(
        args.model,
        dtype=torch.float32,
        device="cpu",
    ).eval()

    environment = versions()
    print(json.dumps(environment, sort_keys=True), file=sys.stderr)
    if args.versions_out is not None:
        args.versions_out.parent.mkdir(parents=True, exist_ok=True)
        args.versions_out.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")

    rows = load_jsonl(args.inputs, args.limit)
    root = args.inputs.parent
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for row in rows:
            audio_path = Path(row["path"])
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            samples, sample_rate = soundfile.read(audio_path, dtype="float32")
            if samples.ndim != 1:
                raise ValueError(f"{audio_path}: expected mono audio")
            waveform = torch.from_numpy(samples).unsqueeze(0)

            chat = ChatState(processor, dtype=torch.float32)
            chat.new_turn("system")
            chat.add_text("Perform ASR.")
            chat.end_turn()
            chat.new_turn("user")
            chat.add_audio(waveform, sample_rate)
            chat.end_turn()
            chat.new_turn("assistant")

            encoded, encoded_lengths = model.conformer(
                chat.audio_in.unsqueeze(0), chat.audio_in_lens
            )
            frame_mask = (
                torch.arange(encoded.shape[-1]).unsqueeze(0)
                < encoded_lengths.unsqueeze(1)
            )
            projected = model.audio_adapter(encoded.mT[frame_mask])

            tokens: list[int] = []
            for generated in model.generate_sequential(
                **chat,
                max_new_tokens=args.max_new_tokens,
                text_temperature=None,
            ):
                if generated.numel() != 1:
                    raise RuntimeError(
                        f"{row['id']}: model switched to audio output during ASR"
                    )
                tokens.append(int(generated.item()))
            transcript_tokens = [token for token in tokens if token not in (7, 130)]
            text = processor.text.decode(
                transcript_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            mel = chat.audio_in.mT.contiguous()
            result = {
                "id": row["id"],
                "mel_frames": mel.shape[0],
                "mel": mel.flatten().tolist(),
                "embeddings": projected.tolist(),
                "tokens": tokens,
                "text": text,
            }
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
            output.flush()
            print(
                f"{row['id']}: mel={mel.shape[0]}, encoder={projected.shape[0]}, "
                f"tokens={len(tokens)}, text={text!r}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
