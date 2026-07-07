from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
DEFAULT_MAX_LENGTH = 8192
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class RerankRequest:
    id: str
    query: str
    documents: list[str]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score rerank requests with the Hugging Face reference ModernBERT reranker."
    )
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to download the model if it is missing from the local Hugging Face cache.",
    )
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args



def load_requests(path: Path) -> list[RerankRequest]:
    rows: list[RerankRequest] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError(f"{path}:{line_no} must decode to an object")
            row_id = data.get("id")
            query = data.get("query")
            documents = data.get("documents")
            if not isinstance(row_id, str):
                raise TypeError(f"{path}:{line_no} id must be a string")
            if not isinstance(query, str):
                raise TypeError(f"{path}:{line_no} query must be a string")
            if not isinstance(documents, list) or not documents:
                raise TypeError(f"{path}:{line_no} documents must be a non-empty list")
            if not all(isinstance(document, str) for document in documents):
                raise TypeError(f"{path}:{line_no} documents entries must all be strings")
            rows.append(RerankRequest(id=row_id, query=query, documents=list(documents)))
    if not rows:
        raise ValueError(f"no rerank requests loaded from {path}")
    return rows



def choose_device(torch_module: Any, requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "mps":
        if not torch_module.backends.mps.is_available():
            raise RuntimeError("--device mps requested, but torch.backends.mps.is_available() is false")
        return "mps"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"



def load_model_and_tokenizer(*, model_name: str, device: str, local_files_only: bool) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return torch, tokenizer, model



def score_requests(
    *,
    requests: list[RerankRequest],
    out_path: Path,
    torch_module: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    max_length: int,
    batch_size: int,
) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for request_index, request in enumerate(requests, start=1):
            request_scores: list[float] = []
            for start in range(0, len(request.documents), batch_size):
                batch_documents = request.documents[start : start + batch_size]
                batch_queries = [request.query] * len(batch_documents)
                encoded = tokenizer(
                    batch_queries,
                    batch_documents,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
                with torch_module.inference_mode():
                    logits = model(**encoded).logits[:, 0]
                request_scores.extend(float(score) for score in logits.detach().cpu().tolist())
                n_pairs += len(batch_documents)
            handle.write(json.dumps({"id": request.id, "scores": request_scores}, ensure_ascii=False))
            handle.write("\n")
            print(
                f"scored {request_index}/{len(requests)} requests ({n_pairs} pairs)",
                file=sys.stderr,
            )
    return {
        "device": device,
        "model": getattr(model, "name_or_path", None),
        "n_requests": len(requests),
        "n_pairs": n_pairs,
        "max_length": max_length,
        "batch_size": batch_size,
    }



def main() -> None:
    args = parse_args()
    requests = load_requests(args.requests)

    import torch

    device = choose_device(torch, args.device)
    torch_module, tokenizer, model = load_model_and_tokenizer(
        model_name=args.model,
        device=device,
        local_files_only=not args.allow_download,
    )
    summary = score_requests(
        requests=requests,
        out_path=args.out,
        torch_module=torch_module,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
