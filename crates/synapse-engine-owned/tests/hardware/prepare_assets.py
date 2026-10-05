"""Copy manifest-pinned cached snapshots, dereferencing links and checking every digest."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
repo = root.parents[3]
manifest = json.loads((repo / "bench/parity/models.json").read_text())
cache = Path.home() / ".cache/huggingface/hub"
for slug, model in manifest["models"].items():
    source = cache / ("models--" + model["hf_repo"].replace("/", "--")) / "snapshots" / model["hf_revision"]
    destination = root / "assets" / slug
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in model["files"].items():
        target = destination / name
        shutil.copyfile(source / name, target, follow_symlinks=True)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"digest mismatch: {slug}/{name}: {actual} != {expected}")
        print(f"{slug}/{name}: {actual}", flush=True)
