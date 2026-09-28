from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from src.dataset.wikipedia.sources import slug
from src.io import write_json
from src.models import digest

MAX_RESPONSE = 8_000_000


class WikipediaRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        target = urlparse(newurl)
        if target.scheme != "https" or target.netloc != "en.wikipedia.org":
            raise ValueError("Refusing Wikipedia redirect to a different origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(source: dict[str, Any], attempts: int = 3) -> dict[str, Any]:
    title = quote(source["title"].replace(" ", "_"), safe="")
    url = f"https://en.wikipedia.org/w/rest.php/v1/page/{title}/with_html"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": os.getenv(
                "WIKIPEDIA_USER_AGENT",
                "QwenCodebaseTrainer/0.1 (local payment engineering research)",
            ),
            "Accept": "application/json",
        },
    )
    opener = urllib.request.build_opener(WikipediaRedirect())
    for attempt in range(attempts):
        try:
            with opener.open(request, timeout=30) as response:
                content = response.read(MAX_RESPONSE + 1)
            if len(content) > MAX_RESPONSE:
                raise ValueError("Wikipedia response exceeds the configured safety limit")
            page = json.loads(content)
            if not isinstance(page.get("html"), str) or not page.get("latest", {}).get("id"):
                raise ValueError("Wikipedia response lacks HTML or a revision ID")
            if (
                not page.get("license", {})
                .get("url", "")
                .startswith("https://creativecommons.org/licenses/by-sa/4.0")
            ):
                raise ValueError("Unexpected Wikipedia content license; inspect before using")
            canonical = "https://en.wikipedia.org/wiki/" + quote(page["key"], safe="()")
            return {
                "schema_version": 1,
                "requested_source": source,
                "page_id": page["id"],
                "title": page["title"],
                "canonical_url": canonical,
                "revision_id": page["latest"]["id"],
                "revision_timestamp": page["latest"].get("timestamp"),
                "fetched_at": dt.datetime.now(dt.UTC).isoformat(),
                "license": page["license"],
                "attribution": f"Wikipedia contributors, {page['title']}, revision {page['latest']['id']}",
                "history_url": canonical + "?action=history",
                "html": page["html"],
                "html_hash": digest(page["html"]),
                "fetch_url": url,
            }
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise RuntimeError(
                    f"Wikipedia fetch failed for {source['title']}: HTTP {exc.code}"
                ) from None
            retry = exc.headers.get("Retry-After", "")
            time.sleep(min(30, int(retry)) if retry.isdigit() else 2**attempt)
        except urllib.error.URLError:
            if attempt + 1 == attempts:
                raise RuntimeError(
                    f"Wikipedia network request failed for {source['title']}"
                ) from None
            time.sleep(2**attempt)
    raise RuntimeError("Wikipedia retries exhausted")


def fetch(sources: list[dict[str, Any]], raw_dir: Path, refresh: bool = False) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source in sources:
        path = raw_dir / f"{slug(source['title'])}.json"
        try:
            if path.exists() and not refresh:
                cached = json.loads(path.read_text())
                if cached.get("requested_source") != source or cached.get("html_hash") != digest(
                    cached.get("html", "")
                ):
                    raise ValueError("Cached source changed or is corrupt; use --refresh")
                status = "cached"
            else:
                write_json(path, download(source))
                status = "fetched"
                time.sleep(0.1)
            results.append({"title": source["title"], "status": status, "path": str(path)})
        except (RuntimeError, ValueError, OSError) as exc:
            results.append({"title": source["title"], "status": "failed", "reason": str(exc)})
    report = {
        "sources": results,
        "successful": sum(r["status"] != "failed" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
    }
    write_json(raw_dir / "fetch-manifest.json", report)
    return report
