"""Fetch four catalog models' CUDA/Vulkan files into a byte-pinned local mirror.

Layout: <mirror>/<hf_repo>/resolve/<revision>/<file>, which the certification
runner serves to the candidate as its hf_endpoint. Every file is checked
against the catalog's size and sha256 before it is kept.
"""
import hashlib, json, os, sys, urllib.request

checkout, mirror = sys.argv[1], sys.argv[2]
catalog = json.load(open(os.path.join(checkout, "crates/synapse-module/src/catalog/models.json")))
for model in catalog["models"]:
    if model['id'] not in {'gte-modernbert-base', 'gte-reranker-modernbert-base', 'qwen3-embedding-0.6b', 'qwen3-reranker-0.6b'}:
        continue
    repo, rev = model["upstream"]["hf_repo"], model["upstream"]["revision"]
    for f in model["files"]:
        if not {'cuda', 'vulkan'}.intersection(f['backends']):
            continue
        dest = os.path.join(mirror, repo, "resolve", rev, f["path"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            url = f"https://huggingface.co/{repo}/resolve/{rev}/{f['path']}"
            urllib.request.urlretrieve(url, dest + ".part")
            os.replace(dest + ".part", dest)
        h = hashlib.sha256()
        with open(dest, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        size = os.path.getsize(dest)
        ok = h.hexdigest() == f["sha256"] and size == f["size_bytes"]
        print(model["id"], f["path"], size, h.hexdigest(), "OK" if ok else "MISMATCH", flush=True)
        if not ok:
            sys.exit(1)
print("MIRROR_DONE")
