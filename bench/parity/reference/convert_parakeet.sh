#!/usr/bin/env bash
# Produce real package pins without returning gigabytes of converted weights.
set -euo pipefail
while :; do echo "parakeet conversion alive $(date -u +%T)"; sleep 20; done &
heartbeat=$!
trap 'kill "$heartbeat" 2>/dev/null || true; wait "$heartbeat" 2>/dev/null || true' EXIT
started=$SECONDS
cargo --version
rustc --version
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

root = Path("bench/parity")
model = json.loads((root / "models.json").read_text())["models"]["parakeet-tdt-0.6b-v3"]
snapshot = root / "target/conversion-hf" / ("models--" + model["hf_repo"].replace("/", "--")) / "snapshots" / model["hf_revision"]
snapshot.mkdir(parents=True, exist_ok=True)
for name, expected in model["files"].items():
    destination = snapshot / name
    source = root / "checkpoints/parakeet-tdt-0.6b-v3" / name
    if source.exists():
        shutil.copyfile(source, destination)
    else:
        url = f'https://huggingface.co/{model["hf_repo"]}/resolve/{model["hf_revision"]}/{name}'
        with urllib.request.urlopen(url, timeout=180) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    with destination.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"checkpoint digest mismatch: {name}")
    print(f"checkpoint {name} {actual}", flush=True)
PY
mkdir -p bench/parity/target/parakeet-packages
for lane in owned-cuda owned-vulkan ane-direct-worker; do
  profile="parakeet-tdt-0.6b-v3.$lane"
  cargo run --release --locked --manifest-path bench/parity/Cargo.toml --bin parity-manifest -- \
    convert --profile "$profile" --hf-cache bench/parity/target/conversion-hf \
    --out "bench/parity/target/parakeet-packages/$profile.safetensors"
done
python3 - "$((SECONDS - started))" <<'PY'
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

root = Path("bench/parity")
model = json.loads((root / "models.json").read_text())["models"]["parakeet-tdt-0.6b-v3"]
checkpoint = root / "target/conversion-hf" / ("models--" + model["hf_repo"].replace("/", "--")) / "snapshots" / model["hf_revision"] / "model.safetensors"

def header(file):
    length = struct.unpack("<Q", file.read(8))[0]
    return json.loads(file.read(length)), length + 8

with checkpoint.open("rb") as original:
    original_header, original_offset = header(original)
    profiles = {}
    for lane in ("owned-cuda", "owned-vulkan", "ane-direct-worker"):
        profile = "parakeet-tdt-0.6b-v3." + lane
        package = root / "target/parakeet-packages" / (profile + ".safetensors")
        with package.open("rb") as file:
            converted, offset = header(file)
            counters = [name for name, value in converted.items() if name != "__metadata__" and value["dtype"] == "I64"]
            if len(counters) != 24 or len(converted) != 724:
                raise RuntimeError("converted Parakeet tensor inventory differs")
            for name in counters:
                source, target = original_header[name], converted[name]
                if source["shape"] != [] or target["shape"] != []:
                    raise RuntimeError("BatchNorm counter must be scalar")
                original.seek(original_offset + source["data_offsets"][0])
                file.seek(offset + target["data_offsets"][0])
                if original.read(8) != file.read(8):
                    raise RuntimeError("BatchNorm integer counter changed during conversion")
            file.seek(0)
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        profiles[profile] = {"bytes": package.stat().st_size, "sha256": digest, "integer_counters_verified": 24}
        print(json.dumps({"profile": profile, **profiles[profile]}, sort_keys=True), flush=True)
receipt = {"checkpoint_sha256": model["files"]["model.safetensors"], "conversion_rule": "v1", "profiles": profiles,
           "cargo_version": subprocess.check_output(["cargo", "--version"], text=True).strip(),
           "rustc_version": subprocess.check_output(["rustc", "--version"], text=True).strip(), "seconds": int(sys.argv[1])}
path = Path(".cortexkit/internal-docs/investigations/s01-reference/package-conversion.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
print("receipt sha256 " + hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
PY
