#!/usr/bin/env python3
"""Generate the fp32 reference data for the Synapse model catalog.

This script is the checked-in generator for two artefacts:

* the ``self_check`` objects of the catalog entries in
  ``crates/synapse-module/src/catalog/models.json``, which the module compares
  a freshly loaded catalog lane against on the user's machine;
* the gte-reranker release evidence corpus
  ``crates/synapse-module/src/fixtures/catalog_rerank_gte_modernbert_fp32.json``
  (at least 20 query/candidate pairs), which the pre-release evidence run
  scores the Metal lane against.

Every reference comes from an fp32 CPU run that is independent of the lanes
under test:

* ``gte-modernbert-base``: Transformers fp32 on CPU, eager attention, the
  hidden state of the first ([CLS]) token, L2-normalised.
* ``gte-reranker-modernbert-base``: Transformers fp32 on CPU, eager attention,
  the single classification logit of each (query, candidate) pair; the stored
  score is ``sigmoid(raw logit)``.
* ``qwen3-embedding-0.6b``: no model run. The 8 inputs and vectors are copied
  from the existing candle-transformers fp32 corpus
  ``crates/synapse-module/src/fixtures/probe_corpus_qwen3_embedding_fp32.json``,
  which was produced at the same pinned revision.

Inputs are read only from the Hugging Face cache snapshot of each entry's
pinned revision, and only the files ``models.json`` declares; each file's
size and sha256 are checked against ``models.json`` before use. The script
refuses to run on library versions other than the pinned ones below, runs
single-threaded with deterministic algorithms and a fixed seed, encodes one
sequence at a time (no padding), and writes its outputs with a fixed layout,
so rerunning it on the pinned versions reproduces the committed files byte
for byte. It rewrites ``models.json`` in place and preserves every field it
does not generate (file digests, backends, fingerprints).

Exact command, from the repository root:

    uv run --no-project --python 3.12 \\
        --with transformers==5.17.0 --with torch==2.14.0 \\
        python bench/catalog/gen_catalog_references.py

Optional environment: ``HF_HUB_CACHE`` (default
``~/.cache/huggingface/hub``) locates the snapshots. Fetch missing snapshots
first with ``huggingface-cli download <repo> --revision <revision>``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REQUIRED_VERSIONS = {"transformers": "5.17.0", "torch": "2.14.0"}

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "crates/synapse-module/src/catalog/models.json"
FIXTURES = REPO_ROOT / "crates/synapse-module/src/fixtures"
QWEN3_CORPUS_PATH = FIXTURES / "probe_corpus_qwen3_embedding_fp32.json"
GTE_CORPUS_PATH = FIXTURES / "probe_corpus_gte_modernbert_ort_fp32.json"
RERANK_CORPUS_PATH = FIXTURES / "catalog_rerank_gte_modernbert_fp32.json"
RERANK_CORPUS_ID = "catalog_rerank_gte_modernbert_fp32"
SEED = 0

# Bumped whenever the self-check inputs or references change, so installed
# lanes re-run their self-check against the new data.
FIXTURE_REVISION = 1

# Items of the shared 64-text probe corpus used as the embed self-check
# inputs: short, long, code, punctuation and mixed-case texts.
EMBED_SELF_CHECK_ITEM_IDS = ["p00", "p14", "p19", "p47", "p49", "p56", "p57", "p59"]

# The rerank self-check: 4 queries, each with candidates whose reference
# scores spread from clearly relevant to clearly irrelevant.
RERANK_SELF_CHECK_INPUTS = [
    {
        "query": "How do I reverse a list in Python?",
        "candidates": [
            "Call list.reverse() to reverse a list in place, or use slicing with lst[::-1] to get a reversed copy.",
            "Python lists are ordered, mutable sequences that can hold items of different types.",
            "The Amazon river flows through Brazil, Peru and Colombia before reaching the Atlantic.",
        ],
    },
    {
        "query": "What causes the seasons on Earth?",
        "candidates": [
            "Seasons happen because Earth's axis is tilted about 23.5 degrees relative to its orbit around the Sun.",
            "The Moon's gravity raises the ocean tides twice a day.",
            "A sourdough starter needs regular feeding with flour and water.",
            "Earth is slightly closer to the Sun in January than in July.",
        ],
    },
    {
        "query": "symptoms of iron deficiency",
        "candidates": [
            "Fatigue, pale skin, shortness of breath and brittle nails are common signs of low iron.",
            "Iron is a chemical element with the symbol Fe and atomic number 26.",
            "The Eiffel Tower is repainted roughly every seven years.",
        ],
    },
    {
        "query": "Which database isolation level prevents phantom reads?",
        "candidates": [
            "The serializable isolation level prevents dirty reads, non-repeatable reads and phantom reads.",
            "Read committed only guarantees that a transaction never sees uncommitted data.",
            "An index speeds up lookups at the cost of extra work on every write.",
            "Bananas are botanically classified as berries.",
        ],
    },
]

# The gte-reranker evidence corpus: 6 queries x 4 candidates = 24 pairs. Each
# query mixes a direct answer, a near miss on the same topic and unrelated
# text, so both the absolute scores and their order within a query are
# exercised.
RERANK_EVIDENCE_ITEMS = [
    {
        "id": "rk00-rust-borrow",
        "query": "Why does the Rust compiler reject two mutable borrows of the same value?",
        "candidates": [
            "Rust allows either one mutable reference or any number of shared references to a value at a time, which rules out data races at compile time.",
            "Cargo is the Rust package manager; it downloads dependencies and builds crates.",
            "The borrow checker tracks lifetimes so that references never outlive the data they point to.",
            "Tomatoes grow best in full sun with consistently moist soil.",
        ],
    },
    {
        "id": "rk01-http-status",
        "query": "what does http status 429 mean",
        "candidates": [
            "429 Too Many Requests means the client has sent too many requests in a given amount of time and is being rate limited.",
            "404 Not Found means the server cannot find the requested resource.",
            "A Retry-After header tells the client how long to wait before making a new request.",
            "The violin has four strings tuned in perfect fifths.",
        ],
    },
    {
        "id": "rk02-photosynthesis",
        "query": "How do plants turn sunlight into chemical energy?",
        "candidates": [
            "In photosynthesis, chlorophyll absorbs light and the plant uses that energy to convert carbon dioxide and water into glucose and oxygen.",
            "Plants absorb water through their roots and move it up through the xylem.",
            "Solar panels convert sunlight directly into electricity using photovoltaic cells.",
            "The stock market closed higher on Friday after a volatile week.",
        ],
    },
    {
        "id": "rk03-sql-join",
        "query": "difference between inner join and left join",
        "candidates": [
            "An inner join returns only rows with matching keys in both tables, while a left join returns every row from the left table and fills unmatched columns with NULL.",
            "A primary key uniquely identifies each row in a table.",
            "A full outer join returns all rows from both tables, matched where possible.",
            "Penguins live almost exclusively in the Southern Hemisphere.",
        ],
    },
    {
        "id": "rk04-sleep",
        "query": "How many hours of sleep does an adult need?",
        "candidates": [
            "Most adults need between seven and nine hours of sleep per night to function well.",
            "Caffeine can stay in the body for several hours and disturb sleep.",
            "Newborn babies sleep up to seventeen hours a day.",
            "The Great Wall of China is thousands of kilometres long.",
        ],
    },
    {
        "id": "rk05-git-rebase",
        "query": "git rebase vs merge",
        "candidates": [
            "Merging creates a merge commit that joins two histories, while rebasing replays your commits on top of another branch to keep a linear history.",
            "git stash temporarily shelves uncommitted changes so you can switch branches.",
            "Interactive rebase lets you reorder, squash or edit commits before sharing them.",
            "A croissant is made from laminated dough layered with butter.",
        ],
    },
]


def fail(message: str) -> None:
    print(f"gen_catalog_references: {message}", file=sys.stderr)
    sys.exit(1)


def check_versions() -> None:
    for package, required in REQUIRED_VERSIONS.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            fail(f"{package}=={required} is required but not installed")
        # torch may carry a local build tag such as "+cpu".
        if installed.split("+", 1)[0] != required:
            fail(f"{package}=={required} is required, found {installed}")


def hub_cache() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_files(entry: dict) -> dict[str, Path]:
    """Map each declared file role to its verified snapshot path."""
    repo = entry["upstream"]["hf_repo"]
    revision = entry["upstream"]["revision"]
    snapshot = hub_cache() / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    by_role: dict[str, Path] = {}
    for file in entry["files"]:
        path = snapshot / file["path"]
        if not path.is_file():
            fail(f"{repo}@{revision}: {file['path']} is missing from {snapshot}")
        size = path.stat().st_size
        if size != file["size_bytes"]:
            fail(f"{repo}@{revision}: {file['path']} is {size} bytes, models.json declares {file['size_bytes']}")
        actual = sha256_file(path)
        if actual != file["sha256"]:
            fail(f"{repo}@{revision}: {file['path']} has sha256 {actual}, models.json declares {file['sha256']}")
        by_role[file["role"]] = path
    for role in ("model", "tokenizer", "config"):
        if role not in by_role:
            fail(f"{entry['id']}: models.json declares no {role} file")
    return by_role


def setup_torch():
    import torch

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def model_dir(files: dict[str, Path], scratch: Path) -> Path:
    """A directory holding only the verified config and weights, so
    Transformers reads nothing the catalog does not pin."""
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "config.json").symlink_to(files["config"])
    (scratch / "model.safetensors").symlink_to(files["model"])
    return scratch


def reference_tool(kind: str) -> str:
    versions = ", ".join(f"{name} {version}" for name, version in REQUIRED_VERSIONS.items())
    return f"{versions}; CPU fp32, eager attention, {kind}; bench/catalog/gen_catalog_references.py"


def gte_embed_self_check(entry: dict, scratch: Path) -> dict:
    torch = setup_torch()
    from transformers import AutoModel

    files = verified_files(entry)
    tokenizer = load_tokenizer(files["tokenizer"])
    model = AutoModel.from_pretrained(
        model_dir(files, scratch / entry["id"]),
        attn_implementation="eager",
        dtype=torch.float32,
    ).eval()
    corpus = {item["id"]: item["text"] for item in json.loads(GTE_CORPUS_PATH.read_text())["items"]}
    inputs = [corpus[item_id] for item_id in EMBED_SELF_CHECK_ITEM_IDS]
    vectors = []
    with torch.no_grad():
        for text in inputs:
            ids = tokenizer.encode(text, add_special_tokens=True).ids
            input_ids = torch.tensor([ids], dtype=torch.long)
            hidden = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).last_hidden_state
            cls = hidden[0, 0].to(torch.float32)
            cls = cls / torch.linalg.vector_norm(cls)
            vectors.append([float(value) for value in cls.tolist()])
    return {
        "fixture_revision": FIXTURE_REVISION,
        "reference_tool": reference_tool("CLS pooling, L2 normalization"),
        "inputs": inputs,
        "reference": {"vectors": vectors},
    }


def sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def rerank_logits(entry: dict, scratch: Path, groups: list[dict]) -> list[list[float]]:
    torch = setup_torch()
    from transformers import AutoModelForSequenceClassification

    files = verified_files(entry)
    tokenizer = load_tokenizer(files["tokenizer"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir(files, scratch / entry["id"]),
        attn_implementation="eager",
        dtype=torch.float32,
    ).eval()
    logits = []
    with torch.no_grad():
        for group in groups:
            row = []
            for candidate in group["candidates"]:
                ids = tokenizer.encode(group["query"], candidate, add_special_tokens=True).ids
                input_ids = torch.tensor([ids], dtype=torch.long)
                output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
                row.append(float(output[0, 0].to(torch.float32).item()))
            logits.append(row)
    return logits


def qwen3_embed_self_check() -> dict:
    corpus = json.loads(QWEN3_CORPUS_PATH.read_text())
    items = {item["id"]: item for item in corpus["items"]}
    chosen = [items[item_id] for item_id in EMBED_SELF_CHECK_ITEM_IDS]
    return {
        "fixture_revision": FIXTURE_REVISION,
        "reference_tool": (
            "candle-transformers 0.10.2; CPU f32, last-token pooling, L2 normalization; items "
            + ",".join(EMBED_SELF_CHECK_ITEM_IDS)
            + " of crates/synapse-module/src/fixtures/probe_corpus_qwen3_embedding_fp32.json"
        ),
        "inputs": [item["text"] for item in chosen],
        "reference": {"vectors": [item["vector"] for item in chosen]},
    }


def dumps(value, indent: int = 0) -> str:
    """JSON with two-space indentation, except that arrays holding only
    scalars (vectors, score rows, file backend lists) stay on one line."""
    pad = "  " * indent
    inner = "  " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        rows = [f"{inner}{json.dumps(key, ensure_ascii=False)}: {dumps(item, indent + 1)}" for key, item in value.items()]
        return "{\n" + ",\n".join(rows) + "\n" + pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in value) + "]"
        rows = [inner + dumps(item, indent + 1) for item in value]
        return "[\n" + ",\n".join(rows) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False)


def write(path: Path, value) -> None:
    path.write_text(dumps(value) + "\n", encoding="utf-8")


def main() -> None:
    check_versions()
    catalog = json.loads(CATALOG_PATH.read_text())
    entries = {entry["id"]: entry for entry in catalog["models"]}
    with tempfile.TemporaryDirectory(prefix="synapse-catalog-refs-") as tmp:
        scratch = Path(tmp)

        entries["gte-modernbert-base"]["self_check"] = gte_embed_self_check(entries["gte-modernbert-base"], scratch)
        entries["qwen3-embedding-0.6b"]["self_check"] = qwen3_embed_self_check()

        reranker = entries["gte-reranker-modernbert-base"]
        self_check_logits = rerank_logits(reranker, scratch / "self-check", RERANK_SELF_CHECK_INPUTS)
        reranker["self_check"] = {
            "fixture_revision": FIXTURE_REVISION,
            "reference_tool": reference_tool("sigmoid(raw logit) of the classification head"),
            "inputs": RERANK_SELF_CHECK_INPUTS,
            "reference": {"scores": [[sigmoid(logit) for logit in row] for row in self_check_logits]},
        }

        evidence_logits = rerank_logits(reranker, scratch / "evidence", RERANK_EVIDENCE_ITEMS)
        corpus = {
            "corpus_id": RERANK_CORPUS_ID,
            "comment": (
                "Release evidence corpus for the gte-reranker-modernbert-base catalog entry. "
                "Each score is sigmoid(raw logit) of the fp32 Transformers reference on CPU."
            ),
            "generation_command": (
                "uv run --no-project --python 3.12 --with transformers==5.17.0 --with torch==2.14.0 "
                "python bench/catalog/gen_catalog_references.py"
            ),
            "reference_tool": reference_tool("sigmoid(raw logit) of the classification head"),
            "upstream": reranker["upstream"],
            "files": [
                {"path": file["path"], "sha256": file["sha256"], "size_bytes": file["size_bytes"]}
                for file in reranker["files"]
            ],
            "pairs": sum(len(item["candidates"]) for item in RERANK_EVIDENCE_ITEMS),
            "items": [
                {
                    "id": item["id"],
                    "query": item["query"],
                    "candidates": item["candidates"],
                    "raw_logits": logits,
                    "scores": [sigmoid(logit) for logit in logits],
                }
                for item, logits in zip(RERANK_EVIDENCE_ITEMS, evidence_logits)
            ],
        }

    write(CATALOG_PATH, catalog)
    write(RERANK_CORPUS_PATH, corpus)
    print(f"wrote {CATALOG_PATH.relative_to(REPO_ROOT)} and {RERANK_CORPUS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
