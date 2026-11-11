#!/usr/bin/env python3
"""Resolve immutable dataset artifacts and seal a deterministic corpus selection.

The corpus records the archive URL and SHA-256 as well as each embedded audio
member's SHA-256. Certification can therefore fetch and verify the same bytes
without relying on expiring dataset-server URLs or unpinned streaming datasets.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

from parakeet_reference import MODEL_REPO, MODEL_REVISION, sha256_file

LIBRI_REV = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
FLEURS_REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
LANGUAGES = ("de_de", "fr_fr", "es_419", "it_it", "pl_pl", "nl_nl", "ru_ru", "el_gr")


def fetch(url, digest, cache):
    destination = cache / digest
    if not destination.exists():
        cache.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha256_file(temporary) != digest:
            temporary.unlink()
            raise RuntimeError(f"SHA-256 mismatch: {url}")
        temporary.rename(destination)
    if sha256_file(destination) != digest:
        raise RuntimeError(f"corrupt cache: {destination}")
    return destination


def dataset_sources(repo, revision, paths):
    metadata = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true"))
    assert metadata["sha"] == revision
    rows = {row["rfilename"]: row for row in metadata["siblings"]}
    sources = {}
    for path in paths:
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
        row = rows[path]
        digest = row.get("lfs", {}).get("sha256")
        if digest is None:
            raw = urllib.request.urlopen(url).read()
            digest = hashlib.sha256(raw).hexdigest()
        sources[path] = {"url": url, "sha256": digest, "bytes": row["size"]}
    return sources


def audio_member(raw, cache):
    import soundfile as sf
    digest = hashlib.sha256(raw).hexdigest()
    path = cache / "audio" / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    info = sf.info(io.BytesIO(raw))
    return {"audio_sha256": digest, "sample_count": info.frames,
            "sample_rate": info.samplerate, "channels": info.channels}


def prepare(cache, output):
    import pyarrow.parquet as pq
    sources, cases = {}, []
    for split in ("clean", "other"):
        path = f"{split}/test/0000.parquet"
        source = dataset_sources("openslr/librispeech_asr", LIBRI_REV, [path])[path]
        key = f"librispeech-test-{split}"
        sources[key] = source
        artifact = fetch(source["url"], source["sha256"], cache / "sources")
        rows = pq.read_table(artifact).to_pylist()
        # Selection is fixed by id, not by model output or transcription accuracy.
        rows.sort(key=lambda row: row["id"])
        selected = rows[:200]
        for row in selected:
            case = {"id": f"{key}-{row['id']}", "utterance_id": row["id"],
                    "split": key, "ground_truth": row["text"], "source": key,
                    "member": row["audio"]["path"]}
            case.update(audio_member(row["audio"]["bytes"], cache))
            cases.append(case)
        print(key, len(selected), flush=True)
    for language in LANGUAGES:
        paths = [f"data/{language}/test.tsv", f"data/{language}/audio/test.tar.gz"]
        pinned = dataset_sources("google/fleurs", FLEURS_REV, paths)
        tsv, archive = (pinned[path] for path in paths)
        key = f"fleurs-{language[:2]}-test"
        sources[key] = archive
        sources[key + "-transcripts"] = tsv
        table = fetch(tsv["url"], tsv["sha256"], cache / "sources")
        # Columns: id, filename, raw transcription, normalized transcription,
        # phonemes, sample count, gender. Keep the original transcription.
        rows = list(csv.reader(table.read_text().splitlines(), delimiter="\t"))
        rows.sort(key=lambda row: (int(row[0]), row[1]))
        selected = rows[:100]
        wanted = {row[1]: row for row in selected}
        artifact = fetch(archive["url"], archive["sha256"], cache / "sources")
        found = {}
        with tarfile.open(artifact, "r:gz") as contents:
            for member in contents:
                name = Path(member.name).name
                if name in wanted:
                    raw = contents.extractfile(member).read()
                    found[name] = (member.name, audio_member(raw, cache))
        for row in selected:
            member, audio = found[row[1]]
            cases.append({"id": f"{key}-{Path(row[1]).stem}", "utterance_id": row[0],
                          "split": key, "ground_truth": row[2], "source": key,
                          "member": member, **audio})
        print(key, len(selected), flush=True)
    corpus = {"schema": "synapse-parakeet-corpus-v1",
              "datasets": {"openslr/librispeech_asr": LIBRI_REV, "google/fleurs": FLEURS_REV},
              "selection": "first 200 LibriSpeech ids per split; first 100 FLEURS (integer id, filename) per language",
              "sources": sources, "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cache, args.output)
