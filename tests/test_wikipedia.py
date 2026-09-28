from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from conftest import source

from src.cli import parser
from src.dataset.wikipedia.clean import clean_article, semantic_chunks
from src.dataset.wikipedia.evidence import extract, safe_text
from src.dataset.wikipedia.fetch import WikipediaRedirect, fetch
from src.dataset.wikipedia.generate import extract_knowledge, make_record
from src.dataset.wikipedia.quality import (
    semantic_deduplicate,
    split_groups,
    validate_record,
)
from src.dataset.wikipedia.scenarios import SCENARIOS
from src.dataset.wikipedia.sources import CATEGORIES, LICENSE_URL, load_sources
from src.models import digest


def article(html: str) -> dict[str, Any]:
    return {
        "html": html,
        "html_hash": digest(html),
        "page_id": 1,
        "revision_id": 12,
        "title": "Idempotence",
        "canonical_url": "https://en.wikipedia.org/wiki/Idempotence",
        "requested_source": {
            "category": "transaction_consistency",
            "tags": ["idempotency"],
            "family": "idempotency",
        },
        "license": {"title": "CC BY-SA 4.0", "url": LICENSE_URL},
        "attribution": "Wikipedia contributors",
        "history_url": "https://en.wikipedia.org/wiki/Idempotence?action=history",
        "fetched_at": "2026-09-28",
    }


def knowledge() -> dict[str, Any]:
    raw = article(
        "<p>An idempotent operation has the same effect when applied repeatedly as when applied once.</p>"
    )
    return extract_knowledge(semantic_chunks(clean_article(raw)))[0]


def local_evidence() -> dict[str, Any]:
    src = source(
        'package p\nfunc refundClientRequestID(refundCode string) string {\nreturn uuid.NewSHA1(uuid.NameSpaceOID, []byte("refund:"+refundCode)).String()\n}\n'
    )
    return extract([src])[0]


def test_catalog_categories_and_attribution() -> None:
    rows = load_sources()
    assert {r["category"] for r in rows} == CATEGORIES
    assert len(rows) >= 40
    assert all(r["license"]["url"] == LICENSE_URL for r in rows)


def test_cleaning_removes_noise_without_rewriting_facts() -> None:
    html = """<nav>navigation</nav><table><tr><td>irrelevant table</td></tr></table>
    <p>A database transaction groups related changes into an atomic unit.<sup class="reference">[1]</sup></p>
    <p>A database transaction groups related changes into an atomic unit.</p>
    <h2>History</h2><p>Historical trivia should not remain in the processed article.</p>
    <h3>Early years</h3><p>Additional historical trivia should not leak through subsections.</p>
    <h2>Guarantees</h2><p>Committed changes survive failures when the promised durability assumptions hold.</p>
    <h2>References</h2><ol><li>Unrelated bibliography entry that is not training evidence.</li></ol>"""
    clean = clean_article(article(html))
    content = " ".join(b["content"] for b in clean["blocks"])
    assert content.count("A database transaction") == 1
    assert "[1]" not in content and "trivia" not in content and "bibliography" not in content
    assert (
        "Committed changes survive failures when the promised durability assumptions hold."
        in content
    )


def test_chunking_preserves_sections_and_sentences() -> None:
    text = "A durable transaction preserves a committed change after a process failure. "
    chunks = semantic_chunks(
        clean_article(
            article(
                "<h2>Atomicity</h2><p>"
                + text * 80
                + "</p><h2>Isolation</h2><p>Concurrent operations require a defined database isolation guarantee.</p>"
            )
        )
    )
    assert len(chunks) > 2
    assert {c["section"] for c in chunks} == {"Atomicity", "Isolation"}
    assert all(c["content"].endswith(".") for c in chunks)
    assert all(c["estimated_tokens"] <= 1000 for c in chunks)
    assert all(c["revision_id"] == 12 for c in chunks)


def test_cleaning_detects_corrupt_cache() -> None:
    raw = article("<p>Original source facts are preserved verbatim.</p>")
    raw["html"] += "changed"
    with pytest.raises(ValueError, match="integrity"):
        clean_article(raw)


def test_ast_detector_does_not_count_comments_or_stub_names() -> None:
    files = [source("package p\n// hmac.New(key) is not executed here.\nfunc Outbox() {\n}\n")]
    assert extract(files) == []


def test_local_patterns_are_concrete_and_do_not_claim_remote_guarantees() -> None:
    evidence = local_evidence()
    assert evidence["pattern"] == "stable_refund_key"
    assert "uuid.NewSHA1" in evidence["snippet"]
    assert "not a provider deduplication guarantee" in evidence["observation"]
    assert evidence["snippet_hash"] == digest(evidence["snippet"])


@pytest.mark.parametrize(
    "text",
    [
        "https://private.example.test/orders",
        "http://10.0.0.1/api",
        "payments.internal",
        'password = "sensitive-value"',
    ],
)
def test_training_privacy_filter(text: str) -> None:
    assert not safe_text(text)


def test_grounding_requires_observation_not_just_a_filename() -> None:
    k, e = knowledge(), local_evidence()
    row = make_record(SCENARIOS[0], k, e)
    assert not validate_record(row, {k["chunk_id"]: k}, {e["id"]: e})
    row["output"] = "Use good engineering practices. " + e["reference"]
    assert validate_record(row, {k["chunk_id"]: k}, {e["id"]: e})


def test_stale_code_and_invented_source_facts_fail() -> None:
    k, e = knowledge(), local_evidence()
    row = make_record(SCENARIOS[0], k, e)
    assert "stale_or_fabricated_code" in validate_record(row, {k["chunk_id"]: k}, {})
    row["metadata"]["source_facts"] = ["Invented provider behavior that is not in Wikipedia."]
    assert "unsupported_source_fact" in validate_record(row, {k["chunk_id"]: k}, {e["id"]: e})


def test_appending_unsupported_provider_claim_is_rejected() -> None:
    k, e = knowledge(), local_evidence()
    row = make_record(SCENARIOS[0], k, e)
    row["output"] += "\nThis provider always guarantees exactly once execution."
    assert "unreviewed_or_unsupported_generated_text" in validate_record(
        row, {k["chunk_id"]: k}, {e["id"]: e}
    )


def test_semantic_dedup_prefers_grounded_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.dataset.wikipedia.quality as quality

    k, e = knowledge(), local_evidence()
    linked = make_record(SCENARIOS[0], k, e)
    generic = make_record(SCENARIOS[0], k, None)

    class FixedEmbedder:
        def __init__(self, model: str) -> None:
            pass

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(quality, "Embedder", FixedEmbedder)
    rows, report = semantic_deduplicate([generic, linked], "test-semantic")
    assert rows == [linked] and len(report["removed"]) == 1


def grouped_rows() -> list[dict[str, Any]]:
    rows = []
    for i in range(10):
        row = make_record(SCENARIOS[0], knowledge(), local_evidence())
        row["id"] = digest(str(i))
        row["metadata"].update(source_page_id=i, family=f"family-{i}", scenario_id=f"scenario-{i}")
        row["metadata"]["code_evidence"][0].update(path=f"module{i}/pay.go", snippet=f"call{i}()")
        rows.append(row)
    return rows


def test_grouped_split_approximately_80_10_10() -> None:
    rows = grouped_rows()
    buckets, _ = split_groups(rows)
    assert {k: len(v) for k, v in buckets.items()} == {"train": 8, "validation": 1, "test": 1}
    assert all(all(r["metadata"]["code_linked"] for r in rs) for rs in buckets.values())
    with pytest.raises(ValueError, match="50%"):
        split_groups(rows, 0.49)


def test_shared_source_or_code_cannot_cross_splits() -> None:
    rows = grouped_rows()
    copy_row = copy.deepcopy(rows[0])
    copy_row["id"] = digest("copy")
    rows.append(copy_row)
    buckets, _ = split_groups(rows)
    locations = [
        name
        for name, records in buckets.items()
        if any(r["metadata"]["source_page_id"] == 0 for r in records)
    ]
    assert len(locations) == 1


def test_fetch_cache_and_failures_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.dataset.wikipedia.fetch as fetch_module

    src = load_sources()[0]
    calls = []

    def fake_download(item: dict[str, Any]) -> dict[str, Any]:
        calls.append(item["title"])
        return {"requested_source": item, "html": "article", "html_hash": digest("article")}

    monkeypatch.setattr(fetch_module, "download", fake_download)
    assert fetch([src], tmp_path)["successful"] == 1
    assert fetch([src], tmp_path)["sources"][0]["status"] == "cached"
    assert len(calls) == 1


def test_fetch_does_not_follow_external_redirects() -> None:
    with pytest.raises(ValueError, match="origin"):
        WikipediaRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.test/page")


def test_cli_keeps_existing_dataset_command() -> None:
    old = parser().parse_args(["dataset", "--limit", "5"])
    assert old.dataset_kind is None and old.limit == 5
    new = parser().parse_args(["dataset", "wikipedia", "build", "--all"])
    assert new.dataset_kind == "wikipedia" and new.wiki_stage == "build" and new.all


def test_history_with_extended_heading_is_removed() -> None:
    cleaned = clean_article(
        article(
            "<h2>History and theory</h2><p>A historical transaction example must be excluded from useful source facts.</p><h2>Algorithm</h2><p>Retry delays increase after repeated failed requests to reduce contention.</p>"
        )
    )
    assert [b["section"] for b in cleaned["blocks"]] == ["Algorithm"]


def test_scenario_rejects_unrelated_queue_evidence() -> None:
    from src.dataset.wikipedia.generate import compatible

    kafka = next(s for s in SCENARIOS if s.key == "kafka_outage")
    assert not compatible(
        kafka, {"pattern": "redis_queue_client", "path": "producer.go", "symbol": "NewProducer"}
    )
    assert compatible(
        kafka, {"pattern": "event_publish", "path": "producer.go", "symbol": "Produce"}
    )


def test_expanded_scenario_catalog_has_distinct_cases_and_evidence_contracts() -> None:
    from collections import Counter

    from src.dataset.wikipedia.generate import SOURCE_PREFERENCES

    counts = Counter(s.family for s in SCENARIOS)
    assert len(SCENARIOS) >= 107
    assert len({s.key for s in SCENARIOS}) == len(SCENARIOS)
    assert len({(s.question, s.premise) for s in SCENARIOS}) == len(SCENARIOS)
    for family in ("event_sourcing", "cqrs", "ddd"):
        assert counts[family] >= 20
        assert family in SOURCE_PREFERENCES
    assert len([s for s in SCENARIOS if s.key.startswith("kafka_")]) >= 21
    for s in SCENARIOS[27:]:
        assert s.patterns and len(s.reasoning.split()) >= 30 and len(s.test.split()) >= 10
        assert s.kind in {v.kind for v in SCENARIOS[:27]}


@pytest.mark.parametrize(
    ("path", "body", "pattern", "family"),
    [
        (
            "internal/eventstore/aggregate_store.go",
            "as.repo.ReadSnapshot(ctx, id, version, agg)",
            "snapshot_read",
            "event_sourcing",
        ),
        (
            "internal/eventstore/aggregate_store.go",
            "txn.CreateSnapshot(ctx, agg)",
            "snapshot_write",
            "event_sourcing",
        ),
        (
            "internal/eventstore/aggregate_store.go",
            "txn.CheckAndUpdateVersion(ctx, agg)",
            "version_check",
            "event_sourcing",
        ),
        (
            "internal/transaction/aggregates/order.go",
            "agg.ApplyChange(agg, event)",
            "domain_event",
            "ddd",
        ),
        (
            "internal/transaction/usecase/project.go",
            "u.projection.Apply(ctx, event)",
            "projection_apply",
            "cqrs",
        ),
        (
            "internal/transaction/usecase/refund.go",
            "u.eventstore.Get(ctx, id, agg)",
            "aggregate_load",
            "cqrs",
        ),
    ],
)
def test_architecture_evidence_is_ast_supported(
    path: str, body: str, pattern: str, family: str
) -> None:
    found = extract([source("package p\nfunc Run() {\n" + body + "\n}\n", path=path)])
    assert len(found) == 1
    assert found[0]["pattern"] == pattern and family in found[0]["families"]
    assert body in found[0]["snippet"]
    # A mentioned call in a comment is not evidence of executing it.
    assert not extract([source("package p\nfunc Run() {\n// " + body + "\n}\n", path=path)])


def test_generated_and_test_helpers_do_not_ground_production_examples() -> None:
    text = "package p\nfunc Run() { u.projection.Apply(ctx, event) }\n"
    assert not extract([source(text, path="internal/mocks/project.go")])
    assert not extract([source(text, path="internal/project_test.go")])
    assert not extract([source("// Code generated by a mock generator\n" + text)])


def test_go_assignment_is_not_an_environment_secret() -> None:
    from src.security.scanner import findings

    assert not findings(
        "err = txn.CreateSnapshot(ctx, agg)\naggPaymentOrderType = reflect.TypeOf(agg)"
    )
    assert any(f["rule"] == "environment_value" for f in findings("PAYMENT_CONFIG=value"))
    assert any(f["rule"] == "environment_value" for f in findings("export PAYMENT_CONFIG=value"))
    assert findings('password = "example-secret"')


def test_snapshot_scenario_cannot_use_arbitrary_save_call() -> None:
    from src.dataset.wikipedia.generate import compatible

    scenario = next(s for s in SCENARIOS if s.key == "es_snapshot_tail")
    assert not compatible(
        scenario, {"pattern": "aggregate_save", "path": "usecase.go", "symbol": "Save"}
    )
    assert compatible(
        scenario,
        {"pattern": "snapshot_read", "path": "aggregate_store.go", "symbol": "getFromSnapshot"},
    )


def test_distinct_calls_in_one_function_have_distinct_evidence_ids() -> None:
    text = "package p\nfunc Run() {\na.CreateOutbox(ctx, payment)\nb.CreateOutbox(ctx, refund)\n}\n"
    evidence = extract([source(text)])
    assert len(evidence) == 2
    assert len({e["id"] for e in evidence}) == 2


def test_keyed_kafka_scenario_requires_keyed_call() -> None:
    from src.dataset.wikipedia.generate import compatible

    scenario = next(s for s in SCENARIOS if s.key == "kafka_key_scope")
    evidence = {
        "pattern": "event_publish",
        "path": "kafka/producer.go",
        "symbol": "Produce",
        "call": "p.Publish",
    }
    assert not compatible(scenario, evidence)
    assert compatible(scenario, {**evidence, "call": "p.ProduceRawWithKey"})
