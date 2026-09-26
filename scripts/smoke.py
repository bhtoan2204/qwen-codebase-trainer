#!/usr/bin/env python3
"""Bounded validation on screened local repositories; no training or teacher calls."""

from __future__ import annotations

import json
from dataclasses import replace

from src.config import Settings
from src.dataset.generator import generate
from src.io import write_json, write_jsonl
from src.retrieval.index import Index
from src.scanner.repositories import scan


def main() -> None:
    settings = Settings()
    settings.prepare()
    files = scan(settings)
    # Spread the sample across repositories instead of selecting a single directory.
    repos = sorted({f.repo for f in files})
    sample = [f for repo in repos for f in [x for x in files if x.repo == repo][:4]]
    if not sample:
        raise ValueError("No safe source files found for smoke validation")
    isolated = replace(settings, data=settings.data / "sample")
    counts = generate(sample, isolated.data)
    with Index(settings.artifacts / "sample-index", settings.embedding) as index:
        first = index.update(sample)
        second = index.update(sample)
        if second["embedded_chunks"] != 0:
            raise AssertionError("Unchanged files were embedded again")
        hits = index.search("how is payment idempotency implemented?", 3)
        if not hits:
            raise AssertionError("Sample query returned no results")
        write_jsonl(settings.artifacts / "sample-search.jsonl", hits)
    summary = {
        "sample_files": len(sample),
        "dataset": counts,
        "index_update": first,
        "repeat_update": second,
        "search_results": len(hits),
    }
    write_json(settings.artifacts / "smoke-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
