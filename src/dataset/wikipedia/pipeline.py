from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.config import Settings
from src.dataset.wikipedia.clean import clean_article, semantic_chunks
from src.dataset.wikipedia.evidence import inspect_code
from src.dataset.wikipedia.fetch import fetch
from src.dataset.wikipedia.generate import extract_knowledge, generate
from src.dataset.wikipedia.quality import (
    semantic_deduplicate,
    split_groups,
    validate_dataset,
    validate_record,
)
from src.dataset.wikipedia.sources import CATALOG, load_sources, slug
from src.io import read_jsonl, write_json, write_jsonl
from src.models import digest

DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"


def add_cli(dataset: argparse.ArgumentParser) -> None:
    nested = dataset.add_subparsers(dest="dataset_kind")
    wiki = nested.add_parser("wikipedia", help="Payment knowledge joined to screened local code")
    stages = wiki.add_subparsers(dest="wiki_stage", required=True)
    for name in ("fetch", "process", "generate", "validate", "build"):
        stage = stages.add_parser(name)
        stage.add_argument("--sources", type=Path, default=CATALOG)
        stage.add_argument("--all", action="store_true", help="Include P1/P2 sources as well as P0")
        stage.add_argument(
            "--title", action="append", help="Select exact catalog titles; repeatable"
        )
        stage.add_argument(
            "--refresh", action="store_true", help="Fetch current revisions even if cached"
        )
        stage.add_argument(
            "--offline", action="store_true", help="Use cached Wikipedia and embedding model only"
        )
        stage.add_argument("--min-code-ratio", type=float, default=0.5)
        stage.add_argument(
            "--dedup-model", default=os.getenv("WIKIPEDIA_DEDUP_MODEL", DEFAULT_EMBEDDING)
        )
        stage.add_argument("--similarity-threshold", type=float, default=0.92)
        stage.add_argument("--per-scenario", type=int, default=2)


def process(sources: list[dict[str, Any]], raw_dir: Path, intermediate: Path) -> dict[str, int]:
    chunks = []
    seen_pages: set[int] = set()
    for source in sources:
        path = raw_dir / f"{slug(source['title'])}.json"
        if not path.is_file():
            raise ValueError(f"Missing cached Wikipedia source: {source['title']}; run fetch first")
        raw = json.loads(path.read_text())
        if raw["requested_source"] != source:
            raise ValueError("Source catalog changed; fetch --refresh before processing")
        if raw["page_id"] in seen_pages:
            continue
        seen_pages.add(raw["page_id"])
        article = clean_article(raw)
        write_json(intermediate / "cleaned" / path.name, article)
        chunks.extend(semantic_chunks(article))
    knowledge = extract_knowledge(chunks)
    write_jsonl(intermediate / "chunks.jsonl", chunks)
    write_jsonl(intermediate / "knowledge.jsonl", knowledge)
    report = {
        "source_pages": len(seen_pages),
        "chunks": len(chunks),
        "useful_knowledge_chunks": len(knowledge),
        "skipped_chunks": len(chunks) - len(knowledge),
    }
    write_json(intermediate / "processing-report.json", report)
    write_json(intermediate / "processing-sources.json", sources)
    return report


def verified_knowledge(settings: Settings, intermediate: Path) -> list[dict[str, Any]]:
    saved = read_jsonl(intermediate / "knowledge.jsonl")
    sources = json.loads((intermediate / "processing-sources.json").read_text())
    chunks = []
    pages = set()
    for source in sources:
        path = settings.data / "raw" / "wikipedia" / f"{slug(source['title'])}.json"
        raw = json.loads(path.read_text())
        if raw["requested_source"] != source:
            raise ValueError("Cached source metadata changed; process the sources again")
        if raw["page_id"] not in pages:
            chunks.extend(semantic_chunks(clean_article(raw)))
            pages.add(raw["page_id"])
    if saved != extract_knowledge(chunks):
        raise ValueError("Processed knowledge differs from raw source evidence; run process again")
    return saved


def build_dataset(
    settings: Settings,
    intermediate: Path,
    model: str,
    minimum: float,
    threshold: float,
    per_scenario: int,
) -> dict[str, Any]:
    if not 0.5 <= minimum <= 1:
        raise ValueError("--min-code-ratio must be between .5 and 1; the 50% floor is mandatory")
    knowledge = verified_knowledge(settings, intermediate)
    if not knowledge:
        raise ValueError("No useful processed knowledge; run fetch/process with more sources")
    evidence = inspect_code(settings)
    candidates, generation = generate(knowledge, evidence, per_scenario)
    lookup = {k["chunk_id"]: k for k in knowledge}
    live = {e["id"]: e for e in evidence}
    rejected = []
    accepted = []
    for row in candidates:
        reasons = validate_record(row, lookup, live)
        if reasons:
            rejected.append({"id": row["id"], "reasons": reasons})
        else:
            accepted.append(row)
    if not accepted:
        raise ValueError("No candidates passed evidence/security checks")
    unique, dedup = semantic_deduplicate(accepted, model, threshold)
    buckets, splitting = split_groups(unique, minimum)
    report = validate_dataset(buckets, knowledge, evidence, model, minimum, threshold)
    write_json(intermediate / "validation-report.json", report)
    write_json(
        intermediate / "generation-report.json",
        {**generation, "rejected": rejected, "deduplication": dedup, **splitting},
    )
    if not report["valid"]:
        raise ValueError(
            "Wikipedia dataset failed quality validation; inspect validation-report.json"
        )
    output = settings.data / "processed"
    artifacts = {}
    for split, rows in buckets.items():
        path = output / f"payment_{split}.jsonl"
        write_jsonl(path, rows)
        write_jsonl(
            output / f"payment_{split}.messages.jsonl",
            (
                {
                    "messages": [
                        {"role": "user", "content": r["instruction"] + "\n\n" + r["input"]},
                        {"role": "assistant", "content": r["output"]},
                    ]
                }
                for r in rows
            ),
        )
        artifacts[path.name] = digest(path.read_text())
    write_json(
        intermediate / "build-manifest.json",
        {
            "outputs": artifacts,
            "semantic_model": model,
            "threshold": threshold,
            "minimum_code_ratio": minimum,
            "report": report,
        },
    )
    # Deliberately leave data/train.jsonl and all existing adapters untouched.
    return report


def validate_existing(
    settings: Settings, intermediate: Path, model: str, minimum: float, threshold: float
) -> dict[str, Any]:
    buckets = {
        split: read_jsonl(settings.data / "processed" / f"payment_{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    knowledge = verified_knowledge(settings, intermediate)
    # Reload code from the repositories rather than trusting saved evidence metadata.
    evidence = inspect_code(settings)
    report = validate_dataset(buckets, knowledge, evidence, model, minimum, threshold)
    for split, rows in buckets.items():
        messages = read_jsonl(settings.data / "processed" / f"payment_{split}.messages.jsonl")
        expected = [
            {
                "messages": [
                    {"role": "user", "content": r["instruction"] + "\n\n" + r["input"]},
                    {"role": "assistant", "content": r["output"]},
                ]
            }
            for r in rows
        ]
        if messages != expected:
            report["errors"].append({"reason": "messages_export_mismatch", "split": split})
    report["valid"] = not report["errors"]
    write_json(intermediate / "validation-report.json", report)
    return report


def run(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    sources = load_sources(args.sources)
    if args.title:
        missing = set(args.title) - {s["title"] for s in sources}
        if missing:
            raise ValueError("Unknown catalog title(s): " + ", ".join(sorted(missing)))
        sources = [s for s in sources if s["title"] in args.title]
    elif not args.all:
        sources = [s for s in sources if s["priority"] == "P0"]
    if not 0.5 <= args.min_code_ratio <= 1:
        raise ValueError("Code-linked ratio must be at least 50%")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    raw = settings.data / "raw" / "wikipedia"
    intermediate = settings.data / "processed" / "wikipedia"
    stage = args.wiki_stage
    result: dict[str, Any] = {}
    if stage in {"fetch", "build"}:
        if args.offline:
            if args.refresh:
                raise ValueError("--offline and --refresh cannot be combined")
            missing_cached = [
                s["title"] for s in sources if not (raw / f"{slug(s['title'])}.json").is_file()
            ]
            if missing_cached:
                raise ValueError("Missing offline sources: " + ", ".join(missing_cached))
        else:
            result = fetch(sources, raw, args.refresh)
            if result["failed"]:
                raise ValueError(
                    f"{result['failed']} Wikipedia fetches failed; see {raw / 'fetch-manifest.json'}"
                )
    if stage in {"process", "build"}:
        result = process(sources, raw, intermediate)
    if stage in {"generate", "build"}:
        result = build_dataset(
            settings,
            intermediate,
            args.dedup_model,
            args.min_code_ratio,
            args.similarity_threshold,
            args.per_scenario,
        )
    if stage == "validate":
        result = validate_existing(
            settings, intermediate, args.dedup_model, args.min_code_ratio, args.similarity_threshold
        )
        if not result["valid"]:
            raise ValueError(
                f"Dataset validation failed; see {intermediate / 'validation-report.json'}"
            )
    return result
