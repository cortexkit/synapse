#!/bin/bash
set -euo pipefail
python3 -m venv /root/hfvenv
/root/hfvenv/bin/pip install -q huggingface-hub==0.34.4
/root/hfvenv/bin/python - <<"PY"
import json
from huggingface_hub import hf_hub_download
models = json.load(open("/root/synapse/bench/parity/models.json"))["models"]
for slug, m in models.items():
    for f in m["files"]:
        p = hf_hub_download(repo_id=m["hf_repo"], revision=m["hf_revision"], filename=f, local_dir=f"/root/weights/{slug}")
        print(slug, p, flush=True)
PY
cd /root/weights && sha256sum */model.safetensors */tokenizer.json
echo WEIGHTS_DONE
