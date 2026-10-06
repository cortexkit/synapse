#!/usr/bin/env python3
"""Generate the committed CPU fp32 parity reference fixtures.

The reference runs once per fixture set, on the CPU in fp32, with the seed
pinned in bench/parity/models.json (`reference_seed`) as its only seed source.
It refuses, before loading or scoring anything, unless the installed
Transformers version equals `reference_transformers_version` and the requested
seed equals `reference_seed`. Verification machines compare against the
committed outputs and need no Python.

Usage:
    uv run --no-project --python 3.12 --with transformers==5.16.1 \
        --with torch==2.14.0 python bench/parity/reference/generate_reference.py \
        --hf-cache ~/.cache/huggingface/hub --model qwen3-reranker-0.6b [--seed 0]

Use --catalog instead of --model to regenerate the catalog self-checks and
the GTE reranker release evidence, keeping their existing inputs. --output-dir
stages either mode under a separate root for comparison before replacement.

Each parity run writes fixtures/<slug>/<fixture set id>.json and records its SHA-256
in fixtures/index.json. The fixture set id names the reference version and
seed, so outputs from a different version or seed can never share an id.

Fixture coverage per model: short inputs, inputs whose token counts (including
special tokens and templates) surround the Apple Neural Engine fixed-shape
boundaries of 128 and 512, one padded batch, and one input of exactly 8192
tokens. Rerank models also get candidate pools of 10 and 100.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

# Set these before importing Torch or Transformers: parallel reductions can
# change fp32 rounding even when deterministic algorithms are enabled.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PARITY_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PARITY_DIR.parent.parent
CATALOG_PATH = Path("crates/synapse-module/src/catalog/models.json")
EVIDENCE_PATH = Path("crates/synapse-module/src/fixtures/catalog_rerank_gte_modernbert_fp32.json")
CATALOG_FIXTURE_REVISION = 2
TORCH_VERSION = "2.14.0"
RUNTIME_SETTINGS = {
    "num_threads": 1,
    "num_interop_threads": 1,
    "deterministic_algorithms": True,
    "attn_implementation": "eager",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
_TORCH_CONFIGURED = False
FIXTURE_REVISION = "ref-v1"
BOUNDARY_LENGTHS = (127, 128, 129, 511, 512, 513)
LONG_LENGTH = 8192


class Refused(Exception):
    """The reference refused to run; nothing was loaded or scored."""


def load_manifest(parity_dir: Path = PARITY_DIR) -> dict:
    return json.loads((parity_dir / "models.json").read_text())


def fixture_set_id(slug: str, transformers_version: str, seed: int) -> str:
    return f"{slug}.{FIXTURE_REVISION}.transformers-{transformers_version}.seed-{seed}"


def preflight(manifest: dict, installed_version: str, seed: int) -> None:
    """Refuse unless the installed Transformers and the seed are the pinned ones."""
    reference = manifest["reference"]
    pinned_version = reference["reference_transformers_version"]
    pinned_seed = reference["reference_seed"]
    if installed_version != pinned_version:
        raise Refused(
            f"installed transformers {installed_version} is not the pinned reference version {pinned_version}"
        )
    if seed != pinned_seed:
        raise Refused(f"seed {seed} is not the pinned reference_seed {pinned_seed}")
    if reference["device"] != "cpu" or reference["dtype"] != "fp32":
        raise Refused("the reference runs only on the CPU in fp32")


# ---------------------------------------------------------------------------
# Deterministic inputs. No randomness: the text is a fixed corpus cycled in a
# fixed order, so the inputs are the same on every run.

SENTENCES = (
    "The owned Vulkan worker selects the first eligible discrete GPU and refuses software devices.",
    "Synapse tokenizes every input in the module, so workers only ever receive token ids.",
    "A converted package stores each tensor in lexicographic order with fixed metadata.",
    "Fresh basil, ripe tomatoes and olive oil make a simple summer pasta sauce.",
    "The river flooded the valley after three days of heavy autumn rain.",
    "Reranking orders candidate passages by how well each one answers the query.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "The Neural Engine runs fixed-shape graphs, so inputs are padded to a ladder rung.",
    "A violin concerto usually has three movements: fast, slow and fast again.",
    "CUDA kernels are compiled to PTX and loaded by the driver at run time.",
    "Bees communicate the direction of food sources through a waggle dance.",
    "Embedding vectors are normalized to unit length before cosine comparison.",
)

QUERY = "How does the Vulkan worker choose which GPU to run on?"


def corpus_words():
    index = 0
    while True:
        sentence = SENTENCES[index % len(SENTENCES)]
        for word in f"[{index}] {sentence}".split(" "):
            yield word
        index += 1


def fit_exact(target: int, compose: Callable[[str], list[int]], prefix_words: int = 0) -> str:
    """Text from the corpus whose composed input is exactly `target` tokens."""
    generator = corpus_words()
    for _ in range(prefix_words):
        next(generator)
    pool = [next(generator) for _ in range(target + 1)]

    def length(count: int) -> int:
        return len(compose(" ".join(pool[:count])))

    # Largest word count whose composition fits. Both tokenizers split on
    # whitespace before subword merging, so the token count never falls when
    # a word is appended, which makes the binary search valid; every word adds
    # at least one token, so target + 1 words always overshoot.
    low, high = 0, len(pool)
    while low < high:
        middle = (low + high + 1) // 2
        if length(middle) <= target:
            low = middle
        else:
            high = middle - 1
    text = " ".join(pool[:low])
    for filler in (" a", " the", " and", " of", "."):
        while len(compose(text)) < target:
            longer = text + filler
            # Stop on a filler that overshoots or merges into the previous
            # token without adding one; the next filler gets a turn.
            if not len(compose(text)) < len(compose(longer)) <= target:
                break
            text = longer
        if len(compose(text)) == target:
            return text
    raise RuntimeError(f"could not reach exactly {target} composed tokens")


# ---------------------------------------------------------------------------
# Composition per model, following the manifest grammar.

def make_composer(model_entry: dict, tokenizer) -> Callable[..., list[int]]:
    grammar = model_entry["grammar"]
    kind = grammar["kind"]

    def encode(text: str, special: bool) -> list[int]:
        return list(tokenizer(text, add_special_tokens=special)["input_ids"])

    if kind == "single_sequence":
        return lambda text: encode(text, True)
    if kind == "pair":
        return lambda query, doc: list(tokenizer(query, doc, add_special_tokens=True)["input_ids"])
    if kind == "template":
        template = grammar["template"]
        prefix = encode(template["prefix"], False)
        suffix = encode(template["suffix"], False)

        def compose(query: str, doc: str) -> list[int]:
            body = template["body_format"].format(instruction=template["instruction"], query=query, doc=doc)
            return prefix + encode(body, False) + suffix

        return compose
    raise ValueError(f"unknown grammar kind {kind}")


def build_cases(model_entry: dict, compose) -> list[dict]:
    """Every fixture case for one model: id, category, text(s) and ids."""
    cases: list[dict] = []
    if model_entry["operation"] == "embed":
        def add(case_id: str, category: str, text: str) -> None:
            cases.append({"id": case_id, "category": category, "text": text, "input_ids": compose(text)})

        for index, text in enumerate([
            "Hello world.",
            SENTENCES[1],
            " ".join(SENTENCES[:4]),
            "Ünïcödé text — with punctuation, digits 12345, and an emoji 🙂.",
        ]):
            add(f"short-{index}", "short", text)
        for length in BOUNDARY_LENGTHS:
            add(f"boundary-{length}", "shape_boundary", fit_exact(length, compose))
        batch = [SENTENCES[i] for i in range(3)] + [fit_exact(n, compose, prefix_words=n) for n in (40, 128, 200)]
        for index, text in enumerate(batch):
            add(f"batch-{index}", "batched", text)
        add(f"long-{LONG_LENGTH}", "long", fit_exact(LONG_LENGTH, compose))
        return cases

    def add_pair(case_id: str, category: str, query: str, doc: str, pool: str | None = None) -> None:
        case = {"id": case_id, "category": category, "query": query, "document": doc,
                "input_ids": compose(query, doc)}
        if pool is not None:
            case["pool"] = pool
        cases.append(case)

    for index, doc in enumerate([SENTENCES[0], SENTENCES[3], SENTENCES[7], "Unrelated."]):
        add_pair(f"short-{index}", "short", QUERY, doc)
    for length in BOUNDARY_LENGTHS:
        add_pair(f"boundary-{length}", "shape_boundary", QUERY,
                 fit_exact(length, lambda doc: compose(QUERY, doc)))
    for index in range(4):
        add_pair(f"batch-{index}", "batched", QUERY, " ".join(SENTENCES[index:index + 1 + index]))
    add_pair(f"long-{LONG_LENGTH}", "long", QUERY, fit_exact(LONG_LENGTH, lambda doc: compose(QUERY, doc)))
    for pool_size in (10, 100):
        for index in range(pool_size):
            doc = f"Candidate {index}: " + " ".join(
                SENTENCES[(index * 5 + offset) % len(SENTENCES)] for offset in range(1 + index % 3)
            )
            add_pair(f"pool{pool_size}-{index:03d}", f"pool_{pool_size}", QUERY, doc, pool=f"pool-{pool_size}")
    return cases


# ---------------------------------------------------------------------------
# Torch backend. Imported lazily so the preflight and its tests need neither
# torch nor transformers.

def configure_torch(torch) -> None:
    """Use single-threaded, deterministic reductions to stabilize fp32 rounding.

    Configure once because Torch forbids resetting interop threads after work
    has started, including when another model is loaded in the same process.
    """
    global _TORCH_CONFIGURED
    if torch.__version__.split("+", 1)[0] != TORCH_VERSION:
        raise Refused(f"torch {TORCH_VERSION} is required, found {torch.__version__}")
    if not _TORCH_CONFIGURED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        _TORCH_CONFIGURED = True


class TorchBackend:
    def __init__(self, snapshot: Path, model_entry: dict, seed: int):
        import torch
        import transformers

        configure_torch(torch)
        torch.manual_seed(seed)
        self.torch = torch
        self.entry = model_entry
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot)
        kind = model_entry["architecture"]["class"]
        kwargs = {"dtype": torch.float32, "attn_implementation": "eager"}
        if model_entry["operation"] == "embed":
            self.model = transformers.AutoModel.from_pretrained(snapshot, **kwargs)
        elif kind == "ModernBertForSequenceClassification":
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(snapshot, **kwargs)
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(snapshot, **kwargs)
        self.model.to("cpu").eval()

    def classifier_logit(self, ids: list[int]) -> float:
        """Catalog evidence retains raw logits as well as double-precision sigmoid scores."""
        torch = self.torch
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
        return float(logits[0, 0])

    def score(self, batch: Sequence[list[int]]) -> list:
        """Outputs for one padded batch: vectors for embed, scores for rerank."""
        torch = self.torch
        grammar = self.entry["grammar"]
        pad = grammar["pad"]["id"]
        width = max(len(ids) for ids in batch)
        input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids in batch], dtype=torch.long)
        mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in batch], dtype=torch.long)
        last = torch.tensor([len(ids) - 1 for ids in batch])
        rows = torch.arange(len(batch))
        with torch.no_grad():
            if self.entry["operation"] == "embed":
                hidden = self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
                pooled = hidden[:, 0] if grammar["pooling"] == "cls" else hidden[rows, last]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                return [[float(v) for v in row] for row in pooled]
            if grammar["readout"]["kind"] == "sigmoid_classifier_logit":
                logits = self.model(input_ids=input_ids, attention_mask=mask).logits[:, 0]
                return [float(v) for v in torch.sigmoid(logits)]
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits[rows, last]
            yes, no = grammar["readout"]["yes"]["id"], grammar["readout"]["no"]["id"]
            pair = torch.stack([logits[:, no], logits[:, yes]], dim=-1)
            return [float(v) for v in torch.softmax(pair, dim=-1)[:, 1]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_dir(hf_cache: Path, model_entry: dict) -> Path:
    repo = model_entry["hf_repo"].replace("/", "--")
    return hf_cache / f"models--{repo}" / "snapshots" / model_entry["hf_revision"]


def load_backend(hf_cache: Path, entry: dict, seed: int, backend_factory=None):
    snapshot = snapshot_dir(hf_cache, entry)
    if backend_factory is None:
        for name, pinned in entry["files"].items():
            actual = sha256_file(snapshot / name)
            if actual != pinned:
                raise Refused(f"{snapshot / name} has SHA-256 {actual}, manifest pins {pinned}")
        backend_factory = TorchBackend
    return backend_factory(snapshot, entry, seed)


def catalog_dumps(value, indent: int = 0) -> str:
    """Keep scalar arrays on one line, matching the catalog's existing JSON layout."""
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        rows = [f"{inner}{json.dumps(key, ensure_ascii=False)}: {catalog_dumps(item, indent + 1)}" for key, item in value.items()]
        return "{\n" + ",\n".join(rows) + "\n" + pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in value) + "]"
        return "[\n" + ",\n".join(inner + catalog_dumps(item, indent + 1) for item in value) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False)


def catalog_embed_ids(entry: dict, compose, text: str) -> list[int]:
    ids = compose(text)
    if entry["grammar"]["pooling"] == "last_non_pad":
        # Last-token pooling reads the hidden state at the final token, so the
        # module's tokenizer ends every catalog input with the model's EOS
        # token; the references must see the same ids. Parity cases instead
        # keep the AutoTokenizer output unchanged and add no tokens.
        eos = entry["grammar"]["terminal_tokens"][-1]["id"]
        if not ids or ids[-1] != eos:
            ids.append(eos)
    return ids


def generate_catalog(manifest: dict, hf_cache: Path, seed: int, version: str,
                     repo_root: Path, output_root: Path, backend_factory=None) -> Path:
    catalog = json.loads((repo_root / CATALOG_PATH).read_text())
    evidence = json.loads((repo_root / EVIDENCE_PATH).read_text())
    tool = (
        f"transformers {version}, torch {TORCH_VERSION}; CPU fp32, eager attention, "
        "seed 0, intra-op/inter-op threads 1, deterministic algorithms, "
        "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1; bench/parity/reference/generate_reference.py --catalog"
    )
    for model in catalog["models"]:
        if "self_check" not in model:
            continue
        entry = manifest["models"][model["id"]]
        if model["upstream"] != {"hf_repo": entry["hf_repo"], "revision": entry["hf_revision"]}:
            raise Refused(f"{model['id']}: catalog and parity upstream pins disagree")
        for file in model["files"]:
            if entry["files"].get(file["path"]) != file["sha256"]:
                raise Refused(f"{model['id']}: catalog and parity file pins disagree")
        backend = load_backend(hf_cache, entry, seed, backend_factory)
        compose = make_composer(entry, backend.tokenizer)
        check = model["self_check"]
        if entry["operation"] == "embed":
            check["reference"] = {"vectors": [
                backend.score([catalog_embed_ids(entry, compose, text)])[0] for text in check["inputs"]
            ]}
            kind = "CLS" if entry["grammar"]["pooling"] == "cls" else "terminal EOS last-token"
            check["reference_tool"] = tool + f"; {kind} pooling, L2 normalization"
        else:
            def score_groups(groups):
                rows = []
                for group in groups:
                    logits = [backend.classifier_logit(compose(group["query"], doc)) for doc in group["candidates"]]
                    rows.append((logits, [1.0 / (1.0 + math.exp(-logit)) for logit in logits]))
                return rows

            check["reference"] = {"scores": [scores for _, scores in score_groups(check["inputs"])]}
            check["reference_tool"] = tool + "; sigmoid(raw classifier logit)"
            for item, (logits, scores) in zip(evidence["items"], score_groups(evidence["items"])):
                item["raw_logits"], item["scores"] = logits, scores
            evidence["reference_tool"] = check["reference_tool"]
            evidence["generation_command"] = (
                f"uv run --no-project --python 3.12 --with transformers=={version} --with torch=={TORCH_VERSION} "
                "python bench/parity/reference/generate_reference.py --catalog --hf-cache ~/.cache/huggingface/hub"
            )
        check["fixture_revision"] = CATALOG_FIXTURE_REVISION
    for relative, document in ((CATALOG_PATH, catalog), (EVIDENCE_PATH, evidence)):
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(catalog_dumps(document) + "\n", encoding="utf-8")
    return output_root / CATALOG_PATH


def run(argv: Sequence[str], *, installed_version: str | None = None,
        backend_factory: Callable[[Path, dict, int], Any] | None = None,
        parity_dir: Path = PARITY_DIR, repo_root: Path = REPO_ROOT) -> Path:
    parser = argparse.ArgumentParser(description="Generate deterministic parity or catalog CPU fp32 references.")
    parser.add_argument("--hf-cache", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model")
    mode.add_argument("--catalog", action="store_true", help="write catalog self-checks and reranker evidence")
    parser.add_argument("--output-dir", type=Path, help="stage outputs under this root instead of overwriting committed files")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    manifest = load_manifest(parity_dir)
    seed = manifest["reference"]["reference_seed"] if args.seed is None else args.seed
    if installed_version is None:
        import transformers

        installed_version = str(transformers.__version__)
    # Nothing is loaded, hashed or scored before this check.
    preflight(manifest, installed_version, seed)

    if args.catalog:
        return generate_catalog(manifest, args.hf_cache, seed, installed_version, repo_root,
                                args.output_dir or repo_root, backend_factory)
    entry = manifest["models"][args.model]
    backend = load_backend(args.hf_cache, entry, seed, backend_factory)
    compose = make_composer(entry, backend.tokenizer)
    cases = build_cases(entry, compose)

    # Batched cases run as one right-padded batch; every other case runs
    # alone, so its reference involves no padding at all.
    batched = [case for case in cases if case["category"] == "batched"]
    outputs = backend.score([case["input_ids"] for case in batched]) if batched else []
    for case, output in zip(batched, outputs):
        case["output"] = output
    for case in cases:
        if case["category"] != "batched":
            case["output"] = backend.score([case["input_ids"]])[0]

    set_id = fixture_set_id(args.model, installed_version, seed)
    document = {
        "fixture_set_id": set_id,
        "model": args.model,
        "operation": entry["operation"],
        "hf_revision": entry["hf_revision"],
        "checkpoint_digest": entry["checkpoint_digest"],
        "tokenizer_digest": entry["tokenizer_digest"],
        "reference": {
            "transformers_version": installed_version,
            "seed": seed,
            "device": "cpu",
            "dtype": "fp32",
            "torch_version": getattr(getattr(backend, "torch", None), "__version__", None),
            **RUNTIME_SETTINGS,
        },
        "cases": cases,
    }
    output_root = args.output_dir or parity_dir
    out_dir = output_root / "fixtures" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{set_id}.json"
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")

    index_path = output_root / "fixtures" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[set_id] = {
        "path": str(path.relative_to(output_root)),
        "sha256": sha256_file(path),
        "model": args.model,
        "transformers_version": installed_version,
        "seed": seed,
    }
    index_path.write_text(json.dumps(dict(sorted(index.items())), indent=2) + "\n")
    return path


def main() -> int:
    try:
        path = run(sys.argv[1:])
    except Refused as refusal:
        print(f"generate_reference: refused: {refusal}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
