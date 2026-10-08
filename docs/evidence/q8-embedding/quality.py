#!/usr/bin/env python3
"""Measure quantization on frozen private pools; publish aggregates only.

Cached vectors and judge identities are private even without their source text.
Keep --private-dir outside version control. No query, candidate, or row ID is
written to the public report. Models and arms run sequentially with four CPU
threads; MPS is optional for this quality experiment, not a speed measurement.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

SCHEMES = ("fp32", "f16", "q8_0", "channel", "w8a8", "gguf")
TASK = "Given a web search query, retrieve relevant code snippets that answer the query"
ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def round_away(x):
    # GGML roundf uses ties away from zero, unlike torch.round's ties to even.
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def q8_0(w):
    """Independent GGML reference: 32 values, f32 division, stored f16 scale."""
    shape = w.shape
    if shape[-1] % 32:
        raise ValueError("Q8_0 requires the row width to be divisible by 32")
    b = w.float().reshape(-1, 32)
    d = b.abs().amax(dim=1, keepdim=True) / 127
    inv = torch.where(d == 0, 0, 1 / d)
    q = round_away(b * inv).clamp(-127, 127)
    return (q * d.half().float()).reshape(shape)


def channel_int8(w):
    scale = w.float().abs().amax(dim=-1, keepdim=True) / 127
    inv = torch.where(scale == 0, 0, 1 / scale)
    q = round_away(w.float() * inv).clamp(-127, 127).to(torch.int8)
    return q, scale.float()


def int8_dot(a, b):
    """Exact int32 accumulation, including on MPS without an int8 GEMM API.

    Each f32 partial has at most 512 products of signed 8-bit values. Its
    absolute sum is bounded by 512*127**2 < 2**24, so every product and addition
    is exactly representable. Convert each partial to int32 before adding it.
    This simulates integer arithmetic, not the speed of a native int8 kernel.
    """
    if a.device.type == "cpu":
        return torch._int_mm(a.contiguous(), b.contiguous())
    acc = torch.zeros((a.shape[0], b.shape[1]), device=a.device, dtype=torch.int32)
    for start in range(0, a.shape[1], 512):
        partial = a[:, start:start + 512].float() @ b[start:start + 512].float()
        acc += partial.to(torch.int32)
    return acc


class W8A8Linear(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        q, s = channel_int8(layer.weight.detach())
        self.register_buffer("q", q.T.contiguous())
        self.register_buffer("scale", s.squeeze(-1))
        self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().float())

    def forward(self, x):
        shape = x.shape[:-1]
        a, s = channel_int8(x.reshape(-1, x.shape[-1]))
        acc = int8_dot(a, self.q)
        y = acc.float() * s * self.scale
        if self.bias is not None:
            y += self.bias
        return y.reshape(*shape, -1).to(x.dtype)


def gguf_name(name):
    if name == "embed_tokens.weight":
        return "token_embd.weight"
    if name == "norm.weight":
        return "output_norm.weight"
    bits = name.split(".")
    if bits[0] != "layers":
        raise ValueError(f"unmapped public model parameter: {name}")
    suffix = ".".join(bits[2:])
    mapping = {
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    return f"blk.{bits[1]}.{mapping[suffix]}"


def install_scheme(model, scheme, gguf_path=None):
    audit = {"matrix_parameters": 0, "linear_layers": 0}
    tensors = {}
    if scheme == "gguf":
        from gguf import GGUFReader
        tensors = {t.name: t for t in GGUFReader(gguf_path).tensors}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if scheme == "gguf":
                from gguf import dequantize
                t = tensors[gguf_name(name)]
                data = dequantize(t.data, t.tensor_type).copy()
                if tuple(data.shape) != tuple(p.shape):
                    raise ValueError(f"GGUF shape mismatch: {name}")
                p.copy_(torch.from_numpy(data))
            elif p.ndim == 2 and scheme in ("q8_0", "channel", "w8a8"):
                audit["matrix_parameters"] += 1
                if scheme == "q8_0":
                    p.copy_(q8_0(p))
                elif scheme == "channel":
                    q, s = channel_int8(p)
                    p.copy_(q.float() * s)
                # W8A8 linears must quantize the original f32 values below.
                elif not any(p is layer.weight for layer in model.modules()
                             if isinstance(layer, torch.nn.Linear)):
                    q, s = channel_int8(p)
                    p.copy_(q.float() * s)
        if scheme == "w8a8":
            for parent in list(model.modules()):
                for key, layer in list(parent.named_children()):
                    if isinstance(layer, torch.nn.Linear):
                        setattr(parent, key, W8A8Linear(layer))
                        audit["linear_layers"] += 1
    # Float scales for W8A8 remain float32. Do not model.half() the buffers.
    for p in model.parameters():
        p.data = p.data.to(torch.float32 if scheme == "fp32" else torch.float16)
    return audit


def audit_gguf(snapshot, gguf_path):
    """Compare every served tensor, not a sample or a self-comparison."""
    from gguf import GGUFReader, GGMLQuantizationType, dequantize, quantize
    from safetensors import safe_open
    report = {"sha256": digest(gguf_path), "q8_tensors": 0, "float_tensors": 0,
              "values": 0, "different_from_fp32_sim": 0,
              "different_from_f16_source_sim": 0, "max_abs_fp32_sim": 0.0,
              "max_abs_f16_source_sim": 0.0, "own_vs_package_max_abs": 0.0,
              "types": {}}
    with safe_open(snapshot / "model.safetensors", framework="pt") as sf:
        for t in GGUFReader(gguf_path).tensors:
            matches = [n for n in sf.keys() if n != "lm_head.weight" and gguf_name(n) == t.name]
            if len(matches) != 1:
                raise ValueError(f"GGUF tensor not mapped uniquely: {t.name}")
            w = sf.get_tensor(matches[0]).float()
            actual = torch.from_numpy(dequantize(t.data, t.tensor_type).copy())
            if tuple(actual.shape) != tuple(w.shape):
                raise ValueError(f"GGUF audit shape mismatch: {t.name}")
            kind = t.tensor_type.name
            report["types"][kind] = report["types"].get(kind, 0) + 1
            if t.tensor_type == GGMLQuantizationType.Q8_0:
                report["q8_tensors"] += 1
                own = q8_0(w)
                package = torch.from_numpy(dequantize(quantize(w.numpy(), t.tensor_type), t.tensor_type))
                report["own_vs_package_max_abs"] = max(report["own_vs_package_max_abs"],
                                                       (own - package).abs().max().item())
                f16_sim = q8_0(w.half().float())
                report["different_from_fp32_sim"] += int((actual != own).sum())
                report["different_from_f16_source_sim"] += int((actual != f16_sim).sum())
                report["max_abs_fp32_sim"] = max(report["max_abs_fp32_sim"], (actual - own).abs().max().item())
                report["max_abs_f16_source_sim"] = max(report["max_abs_f16_source_sim"],
                                                       (actual - f16_sim).abs().max().item())
                report["values"] += w.numel()
            else:
                report["float_tensors"] += 1
                if not torch.equal(actual, w):
                    raise ValueError(f"GGUF non-quantized tensor differs: {t.name}")
    if report["own_vs_package_max_abs"] != 0:
        raise ValueError("Independent Q8_0 simulation disagrees with the gguf package")
    return report


def load_pools(data_root):
    pools = []
    for tool, expected_n, expected_pairs in (("aft", 85, 3613), ("ctx", 92, 3730)):
        rows = json.loads((data_root / f"mason-run-20260928/corpus-{tool}.json").read_text())
        if len(rows) != expected_n or sum(len(x["candidates"]) for x in rows) != expected_pairs:
            raise ValueError("Frozen pool population does not match the September evaluation")
        for row in rows:
            if len({c["id"] for c in row["candidates"]}) != len(row["candidates"]):
                raise ValueError("Duplicate candidate identity")
            for c in row["candidates"]:
                if hashlib.sha256(c["text"].encode()).hexdigest() != c["text_sha256"]:
                    raise ValueError("Frozen candidate text digest mismatch")
            row["sol"] = []
            row["presentations"] = []
            for arm in ("A", "B"):
                prefix = data_root / "jev-2609/sol" / f"{arm}-{row['id']}"
                batch = json.loads(Path(f"{prefix}-batch.json").read_text())
                ranking = json.loads(Path(f"{prefix}-ranking.json").read_text())
                identities = batch["identities"]
                all_ids = ranking["relevant"] + ranking["not_relevant"]
                if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(identities.values()):
                    raise ValueError("Incomplete sol partition")
                ids = all_ids
                if set(ids) != {c["id"] for c in row["candidates"]}:
                    raise ValueError("Sol reference and embedding pool differ")
                row["sol"].append(ranking["relevant"])
                row["presentations"].append(batch["presented_ids"])
            pools.append(row)
    return pools


def prepare(pools, tokenizer, slug, max_length, task):
    sequences, lookup = [], {}
    counts = {"truncated": 0, "rows": 0, "tokens": 0}

    def add(text, query=False):
        if slug.startswith("qwen"):
            if query:
                text = f"Instruct: {task}\nQuery: {text}"
            raw = tokenizer.encode(text, add_special_tokens=False)
            ids = raw[:max_length - 1] + [tokenizer.eos_token_id]
            clipped = len(raw) + 1 > max_length
        else:
            raw = tokenizer.encode(text, add_special_tokens=True)
            ids = raw[:max_length]
            if len(raw) > max_length:
                ids[-1] = tokenizer.sep_token_id
            clipped = len(raw) > max_length
        key = tuple(ids)
        if key not in lookup:
            lookup[key] = len(sequences)
            sequences.append(ids)
            counts["truncated"] += int(clipped)
            counts["rows"] += 1
            counts["tokens"] += len(ids)
        return lookup[key]

    indexes = []
    for p in pools:
        indexes.append((add(p["query"], True), [add(c["text"]) for c in p["candidates"]]))
    return sequences, indexes, counts


def embed(model, seqs, tokenizer, slug, batch_size, device):
    # Length sorting reduces padded work, but keep each pool's original identity order.
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    output = np.empty((len(seqs), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            width = max(len(seqs[i]) for i in idx)
            ids = torch.tensor([seqs[i] + [tokenizer.pad_token_id] * (width - len(seqs[i])) for i in idx],
                               device=device)
            mask = torch.tensor([[1] * len(seqs[i]) + [0] * (width - len(seqs[i])) for i in idx], device=device)
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = h[:, 0] if slug.startswith("gte") else h[torch.arange(len(idx), device=device), mask.sum(1) - 1]
            v = F.normalize(pooled.float(), dim=-1).cpu().numpy()
            if not np.isfinite(v).all() or not np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5):
                raise ValueError("Embedding is non-finite or not normalized")
            output[idx] = v
            if start % (batch_size * 100) == 0:
                print(f"embedded {start}/{len(order)}", flush=True)
    return output


def overlap(order, relevant, n):
    denom = min(10, len(relevant), n)
    return len(set(order[:10]) & set(relevant[:10])) / denom if denom else 0.0


def tau(a, b):
    # Strict complete rankings; exact score ties preserve frozen baseline order.
    if len(a) < 2:
        return None
    ranks = {x: i for i, x in enumerate(b)}
    r = [ranks[x] for x in a]
    inversions = sum(r[i] > r[j] for i in range(len(r)) for j in range(i + 1, len(r)))
    return 1 - 4 * inversions / (len(r) * (len(r) - 1))


def rankings(vectors, pools, indexes):
    orders = []
    for p, (q, docs) in zip(pools, indexes):
        scores = vectors[docs].astype(np.float64) @ vectors[q].astype(np.float64)
        idx = np.argsort(-scores, kind="stable")
        orders.append([p["candidates"][int(i)]["id"] for i in idx])
    return orders


def paired(values):
    a = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(2609)
    samples = a[rng.integers(0, len(a), (10000, len(a)))].mean(axis=1)
    return {"mean": float(a.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}


def aggregate(vectors, pools, indexes):
    orders = {s: rankings(v, pools, indexes) for s, v in vectors.items()}
    result = {}
    for tool in ("aft", "ctx"):
        selected = [i for i, p in enumerate(pools) if p["tool"] == tool]
        unique = sorted({j for i in selected for j in [indexes[i][0], *indexes[i][1]]})
        base_judge = [[overlap(orders["f16"][i], pools[i]["sol"][j], len(orders["f16"][i]))
                       for j in (0, 1)] for i in selected]
        noise = [abs(base_judge[k][0] - base_judge[k][1]) for k in range(len(selected))]
        repeat = [overlap(pools[i]["sol"][1], pools[i]["sol"][0], len(pools[i]["candidates"]))
                  for i in selected if pools[i]["presentations"][0] != pools[i]["presentations"][1]]
        group = {"queries": len(selected), "cosine_unique_vectors": len(unique),
                 "sol_B_vs_A_overlap": float(np.mean(repeat)), "sol_repeat_queries": len(repeat),
                 "f16_judge_mean_abs_A_B": float(np.mean(noise)), "schemes": {}}
        for s, v in vectors.items():
            # Normalize again in float64 to avoid measuring f32 norm-roundoff as quantization loss.
            x, y = v[unique].astype(np.float64), vectors["fp32"][unique].astype(np.float64)
            cos = np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
            stats = {"cosine_fp32": dict(zip(("min", "p1", "p50"), np.quantile(cos, [0, .01, .5]).tolist()))}
            for ref in ("fp32", "f16"):
                ov = [overlap(orders[s][i], orders[ref][i], len(orders[s][i])) for i in selected]
                ts = [tau(orders[s][i], orders[ref][i]) for i in selected]
                stats[f"vs_{ref}"] = {"overlap10": float(np.mean(ov)),
                                        "tau": float(np.mean([t for t in ts if t is not None])),
                                        "tau_queries": sum(t is not None for t in ts)}
            judge = np.array([[overlap(orders[s][i], pools[i]["sol"][j], len(orders[s][i]))
                               for j in (0, 1)] for i in selected])
            stats["sol_overlap10"] = {"A": float(judge[:, 0].mean()), "B": float(judge[:, 1].mean()),
                                       "mean": float(judge.mean())}
            stats["sol_delta_f16"] = paired((judge - np.asarray(base_judge)).mean(axis=1))
            group["schemes"][s] = stats
        result[tool] = group
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / ".local/share/cortexkit/synapse/rerank-eval")
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--models", nargs="+", default=["qwen3-embedding-0.6b", "gte-modernbert-base"])
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 2:
        parser.error("positive batch size and max length >= 2 required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2609)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    pools = load_pools(args.data_root)
    manifest = json.loads((ROOT / "bench/parity/models.json").read_text())["models"]
    from huggingface_hub import snapshot_download, hf_hub_download
    report = {"versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "gguf")},
              "device": args.device, "cpu_threads": 4, "max_length": args.max_length,
              "batch_size": args.batch_size, "task": args.task, "models": {}}
    for slug in args.models:
        entry = manifest[slug]
        snapshot = Path(snapshot_download(entry["hf_repo"], revision=entry["hf_revision"],
                                          local_files_only=not args.allow_download))
        for file, expected in entry["files"].items():
            if digest(snapshot / file) != expected:
                raise ValueError(f"Pinned model file hash mismatch: {slug}/{file}")
        gguf_path = args.gguf
        if slug.startswith("qwen") and "gguf" in args.schemes and gguf_path is None:
            gguf_path = Path(hf_hub_download("Qwen/Qwen3-Embedding-0.6B-GGUF",
                                             "Qwen3-Embedding-0.6B-Q8_0.gguf",
                                             revision="370f27d7550e0def9b39c1f16d3fbaa13aa67728",
                                             local_files_only=not args.allow_download))
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        seqs, indexes, counts = prepare(pools, tokenizer, slug, args.max_length, args.task)
        signature = hashlib.sha256(json.dumps([seqs, args.device, args.batch_size,
                                               report["versions"], entry["hf_revision"]]).encode()).hexdigest()
        vectors, runs = {}, {}
        for scheme in args.schemes:
            if scheme == "gguf" and not slug.startswith("qwen"):
                continue
            identity = signature + (digest(gguf_path) if scheme == "gguf" else "")
            cache = args.private_dir / f"{slug}-{scheme}-{identity}.npz"
            if cache.exists():
                with np.load(cache) as saved:
                    vectors[scheme] = saved["vectors"]
                    runs[scheme] = json.loads(str(saved["metadata"]))
                continue
            print(f"start {slug}/{scheme}: {counts}", flush=True)
            t = time.monotonic()
            model = AutoModel.from_pretrained(snapshot, dtype=torch.float32,
                                              attn_implementation="eager").eval()
            if slug.startswith("gte"):
                model.config.reference_compile = False
            audit = install_scheme(model, scheme, gguf_path)
            model.to(args.device)
            vectors[scheme] = embed(model, seqs, tokenizer, slug, args.batch_size, args.device)
            runs[scheme] = {"elapsed_s": time.monotonic() - t, **audit}
            np.savez(cache, vectors=vectors[scheme], metadata=json.dumps(runs[scheme]))
            del model
            gc.collect()
            if args.device == "mps":
                torch.mps.empty_cache()
        info = {"hf_repo": entry["hf_repo"], "revision": entry["hf_revision"],
                "files": entry["files"], "inputs": counts, "runs": runs}
        if "fp32" in vectors and "f16" in vectors:
            info["quality"] = aggregate(vectors, pools, indexes)
        if gguf_path and slug.startswith("qwen"):
            info["gguf_audit"] = audit_gguf(snapshot, gguf_path)
        report["models"][slug] = info
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("aggregate report written; no private texts or identities exported", flush=True)


if __name__ == "__main__":
    main()
