from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.io import write_json, write_jsonl
from src.models import Chunk, SourceFile, digest
from src.parser.chunks import parse
from src.security.scanner import safe

TASKS = {
    "function": "Explain what this function does, citing the code and distinguishing facts from assumptions.",
    "module": "Explain the responsibility and public surface of this module.",
    "invariants": "Infer invariants enforced by this code. Identify the checks enforcing each invariant.",
    "dependencies": "Identify direct dependencies and distinguish calls from transitive runtime dependencies.",
    "errors": "Explain error propagation and recovery, including any ignored errors.",
    "idempotency": "Explain whether idempotency is enforced. Do not infer guarantees from names alone.",
    "concurrency": "Explain concurrency control and shared-state risks, with evidence.",
    "transactions": "Explain transaction boundaries and which operations are inside or outside them.",
    "data_flow": "Describe input-to-output data flow, including validation and side effects.",
    "conventions": "Describe coding conventions demonstrated here without claiming they are universal.",
    "pattern": "Propose a concrete coding task following a pattern shown here and provide its implementation.",
    "location": "Explain where the demonstrated feature is implemented and cite its file location.",
    "architecture": "Summarize the architecture visible in this evidence and identify what is unknown.",
    "tests": "Propose a focused test task and tests following the visible local conventions. State unverified assumptions.",
}


def record(c: Chunk, task: str, prompt: str, answer: str) -> dict[str, Any]:
    return {
        "id": digest(c.reference + task + prompt + answer),
        "task": task,
        "source": {
            "repo": c.repo,
            "path": c.path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "commit": c.commit,
            "content_hash": digest(c.content),
        },
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
    }


def offline_examples(c: Chunk) -> list[dict[str, Any]]:
    """Only mechanically supported targets; ambiguous reasoning is left to reviewed teachers."""

    if c.language != "go" or c.symbol_type not in {"function", "method", "struct", "interface"}:
        return []

    if len(c.content) > 9000 or len(c.content.splitlines()) < 3:
        return []

    prompt = f"Inspect this local code ({c.reference}):\n```go\n{c.content}\n```\n"
    rows = []

    calls = c.facts.get("calls", [])

    TRIVIAL_CALLS = {
        "fmt.Errorf",
        "errors.New",
        "string",
        "make",
    }

    meaningful_calls = [x for x in calls if x not in TRIVIAL_CALLS]

    include_dependency = (
        len(meaningful_calls) >= 3
        and int(digest(c.reference + ":dependencies")[:8], 16) % 100 < 30
    )

    if include_dependency:
        answer = (
            f"The visible call targets in `{c.reference}` are: "
            + ", ".join(f"`{x}`" for x in meaningful_calls)
            + ". These are syntactic call sites; runtime implementations and transitive dependencies require additional context."
        )

        rows.append(
            record(
                c,
                "dependencies",
                prompt + "List the direct call targets and the limits of this evidence.",
                answer,
            )
        )

    doc = c.facts.get("doc", "")

    if doc:
        answer = (
            f"The author documents `{c.symbol}` as:\n{doc}\n\n"
            f"This is the documented contract at `{c.reference}`, "
            "not independent proof of the implementation's behavior."
        )

        rows.append(
            record(
                c,
                "function",
                prompt + "What contract does the author document for this declaration?",
                answer,
            )
        )

    evidence = {
        "concurrency": [
            x
            for x in c.facts.get("constructs", [])
            if x in {
                "go_statement",
                "select_statement",
                "send_statement",
                "receive_statement",
            }
        ],
        "transactions": [
            x
            for x in calls
            if x.rsplit(".", 1)[-1]
            in {
                "Begin",
                "BeginTx",
                "Commit",
                "Rollback",
                "Transaction",
                "WithTransaction",
            }
        ],
        "errors": [
            x
            for x in calls
            if x in {
                "fmt.Errorf",
                "errors.New",
                "errors.Wrap",
                "errors.Is",
                "errors.As",
            }
        ],
    }

    for task, matches in evidence.items():
        if matches:
            answer = (
                f"At `{c.reference}`, the code contains "
                + ", ".join("`" + m + "`" for m in matches)
                + ". These are observable syntax/call sites. "
                "They alone do not establish end-to-end guarantees; "
                "inspect the surrounding control flow and called implementations before asserting them."
            )

            rows.append(
                record(
                    c,
                    task,
                    prompt
                    + f"Identify explicit {task} constructs or calls. "
                    "What can this evidence establish?",
                    answer,
                )
            )

    return [
        row
        for row in rows
        if all(safe(m["content"]) for m in row["messages"])
    ]

def assignments(
    files: list[SourceFile], chunks: list[Chunk], validation_fraction: float
) -> dict[str, str]:
    """Keep modules and exact duplicate nontrivial declarations in the same connected group."""
    keys = {f"{f.repo}/{f.path}": f"{f.repo}/{Path(f.path).parent}" for f in files}
    parent = {module: module for module in keys.values()}

    def root(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    seen: dict[str, str] = {}
    for c in chunks:
        if len(c.content) < 100 or c.symbol_type in {"package", "import"}:
            continue
        key = keys[f"{c.repo}/{c.path}"]
        fingerprint = digest(" ".join(c.content.split()))
        if fingerprint in seen:
            a, b = sorted((root(key), root(seen[fingerprint])))
            parent[b] = a
        seen[fingerprint] = key
    return {
        key: "validation"
        if int(digest(root(module))[:8], 16) / 2**32 < validation_fraction
        else "train"
        for key, module in keys.items()
    }


def generate(
    files: list[SourceFile], output: Path, validation_fraction: float = 0.15
) -> dict[str, int]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    chunks = [c for f in files for c in parse(f)]
    splits = assignments(files, chunks, validation_fraction)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen: set[str] = set()
    for c in chunks:
        fingerprint = digest(" ".join(c.content.split()))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        rows[splits[f"{c.repo}/{c.path}"]].extend(offline_examples(c))
    provenance: list[dict[str, Any]] = []
    for split, examples in rows.items():
        write_jsonl(output / f"{split}.jsonl", ({"messages": e["messages"]} for e in examples))
        provenance.extend({**e, "split": split} for e in examples)
    write_jsonl(output / "provenance.jsonl", provenance)
    write_json(output / "split-manifest.json", splits)
    counts: dict[str, int] = defaultdict(int)
    for row in provenance:
        counts[row["task"]] += 1
    write_json(
        output / "dataset-report.json",
        {
            "counts": dict(counts),
            "train": len(rows["train"]),
            "validation": len(rows["validation"]),
            "teacher_tasks": TASKS,
            "note": "Offline targets are conservative syntax/documentation facts. Broader reasoning and code/test synthesis require reviewed teacher candidates. Small samples may have an empty split; never move individual samples to fill it.",
        },
    )
    return {split: len(examples) for split, examples in rows.items()}
