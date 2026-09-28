from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from src.config import PROJECT

CATALOG = PROJECT / "datasets/payment_wikipedia_sources.json"
CATEGORIES = {
    "payment_domain",
    "transaction_consistency",
    "distributed_system",
    "messaging",
    "reliability",
    "ledger",
    "reconciliation",
    "security",
}
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def load_sources(path: Path = CATALOG) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    seen: set[str] = set()
    for row in rows:
        required = {
            "title",
            "url",
            "category",
            "priority",
            "tags",
            "license",
            "description",
            "family",
        }
        if not required <= row.keys() or not all(row[k] for k in required):
            raise ValueError(
                "Each Wikipedia source requires complete attribution and classification"
            )
        url = urlparse(row["url"])
        if (
            url.scheme != "https"
            or url.netloc != "en.wikipedia.org"
            or not url.path.startswith("/wiki/")
            or url.query
            or url.fragment
        ):
            raise ValueError("Source URLs must be canonical English Wikipedia article URLs")
        if unquote(url.path[6:]).replace("_", " ") != row["title"]:
            raise ValueError("Source title must match its Wikipedia URL")
        if row["category"] not in CATEGORIES or row["priority"] not in {"P0", "P1", "P2"}:
            raise ValueError("Invalid source category or priority")
        if row["license"]["url"] != LICENSE_URL or not isinstance(row["tags"], list):
            raise ValueError("Invalid license or tags")
        key = slug(row["title"])
        if key in seen:
            raise ValueError("Duplicate source title/filename")
        seen.add(key)
    return sorted(rows, key=lambda r: (r["priority"], r["title"]))
