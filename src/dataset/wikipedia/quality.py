from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.dataset.wikipedia.evidence import safe_text
from src.dataset.wikipedia.scenarios import SCENARIOS
from src.models import digest
from src.retrieval.embeddings import Embedder


class Groups:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def root(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a: int, b: int) -> None:
        a, b = sorted((self.root(a), self.root(b)))
        self.parent[b] = a


def similarity_text(row: dict[str, Any]) -> str:
    # Compare the actual question/premise, not shared citations, code, or answer boilerplate.
    question = row["instruction"].split(" Apply the reasoning to ")[0]
    return question + " " + row["metadata"]["scenario_premise"]


def validate_record(
    row: dict[str, Any], knowledge: dict[str, dict[str, Any]], live: dict[str, dict[str, Any]]
) -> list[str]:
    errors = []
    if not all(
        isinstance(row.get(k), str) and len(row[k].strip()) >= 25
        for k in ("instruction", "input", "output")
    ):
        return ["missing_or_trivial_fields"]
    if not all(safe_text(row[k]) for k in ("instruction", "input", "output")):
        errors.append("sensitive_content")
    meta = row.get("metadata", {})
    source = knowledge.get(meta.get("chunk_id", ""))
    if source is None:
        return errors + ["missing_source"]
    chunk = source["chunk"]
    if meta.get("family") != chunk["family"]:
        errors.append("source_concept_mismatch")
    for field, expected in [
        ("source_title", chunk["title"]),
        ("source_url", chunk["source_url"]),
        ("source_revision", chunk["revision_id"]),
        ("section", chunk["section"]),
        ("license", chunk["license"]),
        ("attribution", chunk["attribution"]),
    ]:
        if meta.get(field) != expected:
            errors.append("source_metadata_mismatch:" + field)
    facts = meta.get("source_facts", [])
    if not facts or any(fact not in chunk["content"] or fact not in row["input"] for fact in facts):
        errors.append("unsupported_source_fact")
    scenario = next((s for s in SCENARIOS if s.key == meta.get("scenario_id")), None)
    if (
        scenario is None
        or meta.get("engineering_interpretation") != scenario.reasoning
        or scenario.reasoning not in row["output"]
    ):
        errors.append("unreviewed_engineering_interpretation")
    if (
        "Hypothetical scenario" not in row["input"]
        or "Engineering interpretation" not in row["output"]
    ):
        errors.append("facts_and_interpretation_not_separated")
    evidence = meta.get("code_evidence", [])
    if meta.get("code_linked") is not bool(evidence):
        errors.append("incorrect_linked_flag")
    for item in evidence:
        from src.dataset.wikipedia.generate import compatible

        if scenario is not None and not compatible(scenario, item):
            errors.append("scenario_pattern_mismatch")
        current = live.get(item.get("id"))
        if current is None or current != item:
            errors.append("stale_or_fabricated_code")
            continue
        if (
            item["snippet"] not in row["input"]
            or item["reference"] not in row["output"]
            or item["observation"] not in row["output"]
        ):
            errors.append("decorative_code_reference")
        if meta.get("family") not in item["families"]:
            errors.append("unrelated_code_pattern")
    # Deterministic generators have a constrained answer grammar; manual/model edits need review.
    if scenario is not None and meta.get("scenario_premise") != scenario.premise:
        errors.append("changed_scenario_premise")
    if not errors and scenario is not None:
        from src.dataset.wikipedia.generate import make_record

        expected_row = make_record(scenario, source, evidence[0] if evidence else None)
        if any(row[k] != expected_row[k] for k in ("instruction", "input", "output")):
            errors.append("unreviewed_or_unsupported_generated_text")
        if any(meta.get(k) != value for k, value in expected_row["metadata"].items()):
            errors.append("generated_provenance_mismatch")
    return errors


def semantic_deduplicate(
    rows: list[dict[str, Any]], model: str, threshold: float = 0.92
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.7 <= threshold <= 1:
        raise ValueError("Semantic threshold must be between .7 and 1")
    # Prefer code-grounded examples over generic variants of the same scenario.
    ordered = sorted(rows, key=lambda r: (not r["metadata"]["code_linked"], r["id"]))
    embedder = Embedder(model)
    vectors = np.asarray(embedder.encode([similarity_text(r) for r in ordered]), dtype=np.float32)
    accepted: list[int] = []
    rejected = []
    for i, row in enumerate(ordered):
        duplicate = next(
            (
                j
                for j in accepted
                if similarity_text(row) == similarity_text(ordered[j])
                or float(vectors[i] @ vectors[j]) >= threshold
            ),
            None,
        )
        if duplicate is None:
            accepted.append(i)
        else:
            rejected.append(
                {
                    "id": row["id"],
                    "duplicate_of": ordered[duplicate]["id"],
                    "cosine": float(vectors[i] @ vectors[duplicate]),
                }
            )
    return [ordered[i] for i in accepted], {
        "model": model,
        "semantic": model != "hashing-v1",
        "threshold": threshold,
        "removed": rejected,
    }


def keys(row: dict[str, Any]) -> set[str]:
    meta = row["metadata"]
    result = {
        f"page:{meta['source_page_id']}",
        f"family:{meta['family']}",
        f"scenario:{meta['scenario_id']}",
    }
    for e in meta["code_evidence"]:
        result.add(f"file:{e['repo']}/{e['path']}")
        result.add("code:" + digest(" ".join(e["snippet"].split())))
    return result


def split_groups(
    rows: list[dict[str, Any]], minimum: float = 0.5
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not 0.5 <= minimum <= 1:
        raise ValueError("Minimum code-linked fraction cannot be below 50%")
    groups = Groups(len(rows))
    seen: dict[str, int] = {}
    for i, row in enumerate(rows):
        for key in keys(row):
            if key in seen:
                groups.join(i, seen[key])
            seen[key] = i
    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows):
        components[groups.root(i)].append(row)
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    targets = {"train": 0.8, "validation": 0.1, "test": 0.1}
    ordered = sorted(components.values(), key=lambda rs: (-len(rs), min(r["id"] for r in rs)))
    if len(ordered) < 3:
        raise ValueError(
            "Fewer than three independent provenance groups; fetch more topics/code evidence rather than leak examples across splits"
        )
    group_map = {}
    for index, component in enumerate(ordered):
        empty = [name for name, bucket in buckets.items() if not bucket]
        # Reserve enough whole groups for nonempty holdouts. Minimize the change in
        # squared distance to target sizes, rather than giving the next huge group
        # to an empty 10% holdout regardless of its size.
        choices = empty if len(ordered) - index == len(empty) else list(buckets)
        split = min(
            choices,
            key=lambda name: (
                (len(buckets[name]) + len(component) - len(rows) * targets[name]) ** 2
                - (len(buckets[name]) - len(rows) * targets[name]) ** 2,
                -targets[name],
                name,
            ),
        )
        group_id = digest("|".join(sorted(r["id"] for r in component)))
        group_map[group_id] = split
        for row in component:
            row["metadata"]["split_group"] = group_id
            row["metadata"]["split"] = split
        buckets[split].extend(component)
    dropped: list[dict[str, Any]] = []
    for name, examples in buckets.items():
        linked = [r for r in examples if r["metadata"]["code_linked"]]
        generic = sorted(
            (r for r in examples if not r["metadata"]["code_linked"]), key=lambda r: r["id"]
        )
        allowed_generic = int(len(linked) * (1 - minimum) / minimum + 1e-9)
        dropped.extend(
            {"id": r["id"], "reason": f"{name}: enforce {minimum:.0%} grounded coverage"}
            for r in generic[allowed_generic:]
        )
        buckets[name] = sorted([*linked, *generic[:allowed_generic]], key=lambda r: r["id"])
        if not buckets[name]:
            raise ValueError(
                f"{name} has no eligible grounded examples; build with more topics/evidence"
            )
    return buckets, {
        "groups": group_map,
        "dropped_for_coverage": dropped,
        "split_policy": "connected source-page/concept-family/code-file/exact-code groups; approximate 80/10/10; never split individual related examples",
    }


def validate_dataset(
    buckets: dict[str, list[dict[str, Any]]],
    knowledge: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    model: str,
    minimum: float = 0.5,
    threshold: float = 0.92,
) -> dict[str, Any]:
    lookup = {k["chunk_id"]: k for k in knowledge}
    live = {e["id"]: e for e in evidence}
    errors: list[dict[str, Any]] = []
    counts = {}
    owners: dict[str, str] = {}
    all_rows = []
    for split, rows in buckets.items():
        linked = sum(bool(r.get("metadata", {}).get("code_linked")) for r in rows)
        counts[split] = {
            "total": len(rows),
            "code_linked": linked,
            "code_linked_fraction": linked / len(rows) if rows else 0,
        }
        if not rows or linked / len(rows) < max(0.5, minimum):
            errors.append({"split": split, "reason": "below_minimum_code_coverage"})
        for row in rows:
            for error in validate_record(row, lookup, live):
                errors.append({"id": row.get("id"), "reason": error})
            for key in keys(row):
                if key in owners and owners[key] != split:
                    errors.append({"id": row["id"], "reason": "cross_split_provenance_leakage"})
                owners[key] = split
            all_rows.append(row)
    _, duplicates = semantic_deduplicate(all_rows, model, threshold)
    if duplicates["removed"]:
        errors.append(
            {"reason": "semantic_or_exact_duplicates", "count": len(duplicates["removed"])}
        )
    if not duplicates["semantic"]:
        errors.append({"reason": "semantic_validation_requires_a_sentence_transformer_model"})
    total = sum(c["total"] for c in counts.values())
    linked_total = sum(c["code_linked"] for c in counts.values())
    return {
        "valid": not errors,
        "minimum_code_linked_fraction": max(0.5, minimum),
        "total": total,
        "code_linked": linked_total,
        "code_linked_fraction": linked_total / total if total else 0,
        "splits": counts,
        "errors": errors,
        "semantic_model": model,
        "limits": "Automated evidence and similarity checks are not a proof of architectural correctness; review representative examples before training.",
    }
