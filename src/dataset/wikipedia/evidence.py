from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from src.config import Settings
from src.io import write_json
from src.models import Chunk, SourceFile, digest
from src.parser.chunks import parse
from src.scanner.repositories import scan
from src.security.scanner import safe

# This stage is stricter than the general retrieval scanner: no URLs from proprietary code.
PRIVATE_DATA = re.compile(
    r"https?://\S+|\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[\w.-]+\.(?:internal|local|corp|lan)\b", re.I
)


def safe_text(text: str) -> bool:
    return safe(text) and not PRIVATE_DATA.search(text)


def detect(c: Chunk) -> list[tuple[str, list[str], str, str]]:
    """Return AST-supported observations, not guarantees inferred from symbol names."""
    if c.language != "go" or c.symbol_type not in {"function", "method"}:
        return []
    calls = c.facts.get("calls", [])
    findings = []
    for call in calls:
        leaf = call.rsplit(".", 1)[-1]
        if leaf == "ApplyChange" and "/aggregates/" in c.path:
            findings.append(
                (
                    "domain_event",
                    ["event_sourcing", "ddd"],
                    call,
                    f"`{c.symbol}` invokes `{call}` with a domain event. This identifies a command-to-event boundary, not proof of invariant enforcement or side-effect-free replay.",
                )
            )
        elif call.endswith("projection.Apply"):
            findings.append(
                (
                    "projection_apply",
                    ["cqrs", "event_sourcing"],
                    call,
                    f"`{c.symbol}` dispatches an event through `{call}`. The dispatch does not prove projector error propagation, idempotency, or current read-model state.",
                )
            )
        elif (
            leaf in {"ReadSnapshot", "CreateSnapshot", "CheckAndUpdateVersion"}
            and "/eventstore/" in c.path
        ):
            pattern = {
                "ReadSnapshot": "snapshot_read",
                "CreateSnapshot": "snapshot_write",
                "CheckAndUpdateVersion": "version_check",
            }[leaf]
            findings.append(
                (
                    pattern,
                    ["event_sourcing"],
                    call,
                    f"`{c.symbol}` calls `{call}` in the event-store path. Inspect the repository and surrounding transaction before asserting snapshot consistency or atomic version enforcement.",
                )
            )
        elif leaf == "Get" and "eventstore" in call.lower():
            findings.append(
                (
                    "aggregate_load",
                    ["event_sourcing", "cqrs", "ddd"],
                    call,
                    f"`{c.symbol}` loads an aggregate with `{call}`. The call is an investigation point for reconstruction and command validation, not proof of freshness or concurrency safety.",
                )
            )
        elif leaf == "CreateOutbox":
            findings.append(
                (
                    "outbox_write",
                    ["outbox", "atomicity", "messaging", "consistency"],
                    call,
                    f"`{c.symbol}` calls `{call}` to request an outbox write. This excerpt does not establish that the business update and outbox share one database commit.",
                )
            )
        elif leaf in {"ProduceRawWithKey", "Publish", "Produce"}:
            findings.append(
                (
                    "event_publish",
                    ["messaging", "outbox", "consistency", "retry"],
                    call,
                    f"`{c.symbol}` calls `{call}` to publish data. The call site alone does not establish delivery, consumer deduplication, or transactional coupling to a database.",
                )
            )
        elif call == "hmac.New":
            findings.append(
                (
                    "hmac_construction",
                    ["authentication", "replay", "security"],
                    call,
                    f"`{c.symbol}` constructs a keyed MAC with `hmac.New`. MAC construction alone is not evidence that incoming requests are verified or that old signed requests are rejected.",
                )
            )
        elif call == "uuid.NewSHA1" and "refundCode" in c.content:
            findings.append(
                (
                    "stable_refund_key",
                    ["idempotency", "retry", "payment"],
                    call,
                    f"`{c.symbol}` derives a name-based UUID using `uuid.NewSHA1` and `refundCode`. The same namespace/name input is deterministic; this demonstrates local request identity construction, not a provider deduplication guarantee.",
                )
            )
        elif leaf == "Save" and "eventstore" in call.lower():
            findings.append(
                (
                    "aggregate_save",
                    [
                        "concurrency",
                        "atomicity",
                        "consistency",
                        "callbacks",
                        "reconciliation",
                        "payment",
                    ],
                    call,
                    f"`{c.symbol}` calls `{call}` to persist an aggregate. A call named Save does not prove a version comparison, a database lock, or exactly-once processing; the event-store implementation must be checked.",
                )
            )
        elif leaf == "RecordPartnerCallBack":
            findings.append(
                (
                    "callback_record",
                    ["callbacks", "idempotency", "concurrency", "payment"],
                    call,
                    f"`{c.symbol}` delegates callback state recording to `{call}`. The observed delegation is not by itself proof of a durable duplicate-event check or safe concurrent state transitions.",
                )
            )
        elif call == "asynq.NewClient":
            findings.append(
                (
                    "redis_queue_client",
                    ["resilience", "retry", "messaging"],
                    call,
                    f"`{c.symbol}` constructs an `asynq.NewClient` for background work. Queue-client construction is not a durable payment invariant or a guarantee of successful enqueueing during an outage.",
                )
            )
        elif leaf in {
            "QueryRefundStatus",
            "QueryPaymentStatus",
            "QueryOrderStatus",
            "CheckRefundStatus",
        }:
            findings.append(
                (
                    "provider_query",
                    ["reconciliation", "consistency", "settlement", "payment"],
                    call,
                    f"`{c.symbol}` invokes `{call}` to query status. This supplies a place to investigate reconciliation; it does not prove a periodic reconciler or an authoritative ledger update exists.",
                )
            )
        elif leaf == "ProcessResultData":
            findings.append(
                (
                    "callback_decode",
                    ["callbacks", "authentication", "replay", "security"],
                    call,
                    f"`{c.symbol}` passes callback input to `{call}`. Inspect that implementation before claiming signature validation, timestamp checks, or replay protection.",
                )
            )
    return findings


def extract(files: list[SourceFile]) -> list[dict[str, Any]]:
    evidence = []
    seen: set[tuple[str, str]] = set()
    for source in files:
        if source.language != "go" or (
            source.path.endswith("_test.go")
            or "/mocks/" in source.path
            or "Code generated" in source.content[:500]
        ):
            continue
        for c in parse(source):
            if not safe_text(c.content):
                continue
            for pattern, families, call, observation in detect(c):
                lines = c.content.splitlines()
                # AST establishes the call exists; locate a small verbatim evidence window.
                positions = [
                    i
                    for i, line in enumerate(lines)
                    if call + "(" in line and not line.lstrip().startswith("//")
                ]
                if not positions:
                    continue
                position = positions[0]
                start, end = max(0, position - 2), min(len(lines), position + 5)
                snippet = "\n".join(lines[start:end])
                fingerprint = (pattern, digest(" ".join(snippet.split())))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                evidence.append(
                    {
                        "id": digest(c.id + pattern + call + digest(snippet)),
                        "repo": c.repo,
                        "path": c.path,
                        "symbol": c.symbol,
                        "start_line": c.start_line + start,
                        "end_line": c.start_line + end - 1,
                        "commit": c.commit,
                        "file_hash": source.content_hash,
                        "snippet_hash": digest(snippet),
                        "snippet": snippet,
                        "pattern": pattern,
                        "families": families,
                        "call": call,
                        "observation": observation,
                        "reference": f"{c.repo}/{c.path}:{c.start_line + start}-{c.start_line + end - 1}",
                    }
                )
    return sorted(evidence, key=lambda e: (e["pattern"], e["repo"], e["path"], e["start_line"]))


def inspect_code(settings: Settings) -> list[dict[str, Any]]:
    # Keep the Wikipedia security report separate from the existing retrieval artifacts.
    isolated = replace(settings, artifacts=settings.artifacts / "wikipedia")
    files = scan(isolated)
    evidence = extract(files)
    write_json(
        isolated.artifacts / "code-inventory.json",
        {
            "accepted_files": len(files),
            "evidence_count": len(evidence),
            "patterns": sorted({e["pattern"] for e in evidence}),
            "repositories": sorted({e["repo"] for e in evidence}),
            "evidence": evidence,
            "scope": "Screened eligible source only; unobserved patterns are not proven absent from the repositories.",
        },
    )
    return evidence
