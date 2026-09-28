from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup

from src.models import digest

EXCLUDED_HEADINGS = {
    "references",
    "notes",
    "citations",
    "bibliography",
    "external links",
    "further reading",
    "see also",
    "history",
    "etymology",
}


def clean_article(raw: dict[str, Any]) -> dict[str, Any]:
    """Structural removal only: no paraphrasing, factual rewrites or model calls."""
    if digest(raw["html"]) != raw["html_hash"]:
        raise ValueError("Raw HTML integrity check failed")
    soup = BeautifulSoup(raw["html"], "html.parser")
    for node in soup.select(
        "script, style, nav, footer, table, .navbox, .sidebar, .infobox, .metadata, .ambox, .hatnote, .reflist, .references, .mw-editsection, .noprint, sup.reference"
    ):
        node.decompose()
    for node in soup.select("math"):
        node.replace_with(str(node.get("alttext", node.get_text(" ", strip=True))))
    blocks: list[dict[str, str]] = []
    heading = "Overview"
    suppressed_at: int | None = None
    seen: set[str] = set()
    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"]):
        text = re.sub(
            r"\[\s*(?:\d+(?:\s*[,–-]\s*\d+)*|citation needed|edit)\s*\]",
            "",
            node.get_text(" ", strip=True),
        )
        text = re.sub(r"\s+", " ", text).strip()
        if re.fullmatch(r"h[1-6]", node.name):
            level = int(node.name[1])
            if suppressed_at is not None and level > suppressed_at:
                continue
            suppressed_at = (
                level
                if (text.lower() in EXCLUDED_HEADINGS or text.lower().startswith("history "))
                else None
            )
            heading = text
            continue
        if suppressed_at is not None or node.find_parent(["li", "pre"]) is not None or not text:
            continue
        fingerprint = digest(text.casefold())
        if fingerprint in seen or len(text.split()) < 5:
            continue
        seen.add(fingerprint)
        blocks.append({"section": heading, "content": text})
    return {k: v for k, v in raw.items() if k != "html"} | {"blocks": blocks, "cleaning_version": 1}


def estimated_tokens(text: str) -> int:
    # Conservative language-agnostic estimate, not the Qwen tokenizer.
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)) * 4 // 3)


def semantic_chunks(
    article: dict[str, Any], target_min: int = 300, maximum: int = 1000
) -> list[dict[str, Any]]:
    if not 1 <= target_min <= maximum:
        raise ValueError("Invalid chunk token limits")
    groups: list[tuple[str, list[str]]] = []
    for block in article["blocks"]:
        if not groups or groups[-1][0] != block["section"]:
            groups.append((block["section"], []))
        groups[-1][1].append(block["content"])
    result = []
    for section, paragraphs in groups:
        units: list[str] = []
        for paragraph in paragraphs:
            # Split oversized paragraphs only at sentence boundaries, never by characters.
            units.extend(
                re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
                if estimated_tokens(paragraph) > maximum
                else [paragraph]
            )
        batches: list[list[str]] = []
        current: list[str] = []
        for unit in units:
            if current and estimated_tokens(" ".join([*current, unit])) > maximum:
                batches.append(current)
                current = []
            current.append(unit)
        if current:
            if batches and estimated_tokens(" ".join([*batches[-1], *current])) <= maximum:
                batches[-1].extend(current)
            else:
                batches.append(current)
        for part, batch in enumerate(batches, 1):
            content = "\n\n".join(batch)
            source = article["requested_source"]
            result.append(
                {
                    "id": digest(
                        f"{article['page_id']}:{article['revision_id']}:{section}:{part}:{content}"
                    ),
                    "source": "wikipedia",
                    "title": article["title"],
                    "section": section,
                    "category": source["category"],
                    "tags": source["tags"],
                    "family": source["family"],
                    "content": content,
                    "source_url": article["canonical_url"],
                    "revision_id": article["revision_id"],
                    "page_id": article["page_id"],
                    "license": article["license"],
                    "attribution": article["attribution"],
                    "history_url": article["history_url"],
                    "fetched_at": article["fetched_at"],
                    "estimated_tokens": estimated_tokens(content),
                    "size_note": "short semantic section"
                    if estimated_tokens(content) < target_min
                    else "oversized indivisible sentence"
                    if estimated_tokens(content) > maximum
                    else "target range",
                    "cleaning": "navigation/tables/citations/backmatter/history removed; whitespace normalized; no factual rewrites",
                }
            )
    return result
