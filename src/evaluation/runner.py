from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.io import read_jsonl, write_json, write_jsonl
from src.models import digest
from src.retrieval.index import Index


def questions(provenance: Path, output: Path, limit: int = 30) -> int:
    rows = []
    seen = set()
    for row in read_jsonl(provenance):
        if row["split"] != "validation" or row["task"] != "location":
            continue
        source = row["source"]
        file = f"{source['repo']}/{source['path']}"
        if file in seen:
            continue
        seen.add(file)
        rows.append(
            {
                "id": digest(row["messages"][0]["content"]),
                "question": row["messages"][0]["content"],
                "expected_files": [file],
                "source_hash": source["content_hash"],
                "expected_reference": f"{file}:{source['start_line']}-{source['end_line']}",
            }
        )
        if len(rows) >= limit:
            break
    if not rows:
        raise ValueError("No held-out location examples; generate a larger dataset first")
    write_jsonl(output, rows)
    return len(rows)


def retrieval_evaluation(index: Index, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("Evaluation cases must not be empty")
    rows = []
    for case in cases:
        begin = time.perf_counter()
        hits = index.search(case["question"], 5)
        expected = set(case["expected_files"])
        retrieved = {f"{h['repo']}/{h['path']}" for h in hits}
        rows.append(
            {
                "id": case["id"],
                "source_retrieval_accuracy": len(expected & retrieved) / len(expected),
                "latency_seconds": time.perf_counter() - begin,
                "retrieved_files": sorted(retrieved),
            }
        )
    return {
        "mode": "retrieval-only",
        "cases": rows,
        "mean_recall_at_5": sum(r["source_retrieval_accuracy"] for r in rows) / len(rows),
        "note": "Held-out location questions only; no answer-quality or fine-tuning improvement claim.",
    }


def compare(
    model_name: str, adapter: str, index: Index, cases: list[dict[str, Any]], output: Path
) -> None:
    import gc

    import torch

    from src.inference.qwen import Qwen

    if not cases:
        raise ValueError("Evaluation cases must not be empty")
    rows = []
    for tuned in (False, True):
        model = Qwen(model_name, adapter if tuned else None)
        for rag in (False, True):
            variant = ("D" if rag else "C") if tuned else ("B" if rag else "A")
            for case in cases:
                begin = time.perf_counter()
                hits = index.search(case["question"], 5) if rag else []
                answer = model.answer(case["question"], hits)
                expected = set(case["expected_files"])
                retrieved = {f"{h['repo']}/{h['path']}" for h in hits}
                rows.append(
                    {
                        "id": case["id"],
                        "variant": variant,
                        "question": case["question"],
                        "answer": answer,
                        "expected_files": sorted(expected),
                        "sources": hits,
                        "latency_seconds": time.perf_counter() - begin,
                        "source_retrieval_accuracy": len(expected & retrieved) / len(expected)
                        if rag
                        else None,
                        "file_location_accuracy": sum(file in answer for file in expected)
                        / len(expected),
                        "review": {
                            "reviewer": "",
                            "supported_claims": None,
                            "unsupported_claims": None,
                            "contradicted_claims": None,
                        },
                    }
                )
                write_jsonl(output, rows)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize_reviewed(path: Path, output: Path) -> dict[str, Any]:
    """Claim-level human labels avoid passing word overlap off as groundedness."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        groups.setdefault(row["variant"], []).append(row)
    result = {}
    for variant, rows in groups.items():
        supported = unsupported = contradicted = reviewed = 0
        for row in rows:
            review = row.get("review", {})
            counts = [
                review.get(k)
                for k in ("supported_claims", "unsupported_claims", "contradicted_claims")
            ]
            if review.get("reviewer") and all(type(v) is int and v >= 0 for v in counts):
                supported += counts[0]
                unsupported += counts[1]
                contradicted += counts[2]
                reviewed += 1
        total = supported + unsupported + contradicted
        result[variant] = {
            "cases": len(rows),
            "reviewed_cases": reviewed,
            "answer_groundedness": supported / total if total else None,
            "hallucination_rate": (unsupported + contradicted) / total if total else None,
            "file_location_accuracy": sum(r["file_location_accuracy"] for r in rows) / len(rows),
            "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows),
            "source_retrieval_accuracy": sum(r["source_retrieval_accuracy"] for r in rows)
            / len(rows)
            if all(r["source_retrieval_accuracy"] is not None for r in rows)
            else None,
        }
    report = {
        "variants": result,
        "note": "Groundedness/hallucination use manually reviewed claims only; null means unmeasured. Latency excludes model loading. Location accuracy is literal expected-file mention. Use paired cases and inspect examples before claiming improvement.",
    }
    write_json(output, report)
    return report
