# ANE multi-function package spike

These scripts measure whether one multi-function Core ML package holding the
gte-modernbert-base 128/256/512 buckets can replace the three single-bucket
packages. The results are in `docs/evidence/ane-coreml-multifunction/README.md`.

Artifacts (packages, compiled bundles, vectors, raw reports) go under
`~/.local/share/cortexkit/synapse/ane-multifunction/`, not into git.

```sh
root=~/.local/share/cortexkit/synapse/ane-multifunction
uv venv --python 3.12 "$root/venv"
uv pip install --python "$root/venv/bin/python" -r bench/spikes/ane-minilm/requirements.txt
"$root/venv/bin/python" bench/spikes/ane-multifunction/build.py      # singles + multi-function packages
bench/spikes/ane-multifunction/build_probe.sh                         # Swift Core ML probe
"$root/venv/bin/python" bench/spikes/ane-multifunction/run.py \
  --stages rows compile placement embed reference parity load latency memory evidence
```

`run.py` waits for an empty `campaign_rig_claim` and a 1-minute load average of
at most 16 before every Neural Engine probe. It needs the installed production
bundles under `~/.local/share/cortexkit/models/ane-coreml/` as the baseline, and
the `Alibaba-NLP/gte-modernbert-base` snapshot in the local Hugging Face cache.
