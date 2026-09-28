from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from src.dataset.wikipedia.evidence import safe_text
from src.dataset.wikipedia.scenarios import SCENARIOS, STATES, Scenario
from src.models import digest

ENGINEERING_WORDS = re.compile(
    r"\b(?:aggregate|domain|event|command|query|projection|snapshot|payment|transaction|database|operation|request|message|state|failure|concurren\w*|consisten\w*|signature|authentication|account\w*|ledger|refund|queue|retry|idempoten\w*|publish\w*|attack|settlement|encrypt\w*)\b",
    re.I,
)


PATTERN_APPLICATIONS = {
    "domain_event": "Trace this ApplyChange call into event application and aggregate invariants. Distinguish command validation from replay; check persisted event fields and side effects before claiming the model enforces a rule.",
    "projection_apply": "Follow projection.Apply from the deserialized event into each projector and its persistence/checkpoint boundary. Verify error propagation and version handling; dispatch alone proves neither freshness nor exactly-once projection.",
    "snapshot_read": "Trace ReadSnapshot and the subsequent event-tail read. Compare aggregate identity, snapshot version and target stream version; test fallback using the same committed history before trusting cached state.",
    "snapshot_write": "Inspect CreateSnapshot inside its surrounding save transaction. Check the serialized state and version match the committed prefix; an observed snapshot call alone does not prove atomic publication or restore correctness.",
    "version_check": "Follow CheckAndUpdateVersion into the repository transaction and append operations. Verify stale writers are rejected atomically and rollback does not leave in-memory aggregate bookkeeping suitable for blind reuse.",
    "aggregate_load": "Follow eventstore.Get into reconstruction and its command caller. Verify complete ordered history and version checks at save; reading an aggregate alone does not establish a consistent subsequent write.",
    "stable_refund_key": "Keep the same refundCode for retries of one refund: changing that input changes the name-based identity. Persist the amount/currency binding and result outside this helper; do not treat UUID construction as proof of remote deduplication.",
    "outbox_write": "Follow the shown CreateOutbox call into its repository and transaction context. Verify it commits with the payment mutation; if it runs in a later projection transaction, explicitly test the crash gap and replay behavior rather than assuming the name makes the write atomic.",
    "event_publish": "Trace the actual producer call's errors and its caller's acknowledgement policy. Add durable publication intent before accepting a business operation; a keyed send does not show that a failed send will be retried or that the consumer deduplicates.",
    "hmac_construction": "Trace the bytes passed to hmac.New and the subsequent MAC calculation into the caller. Check whether the receiver verifies the raw payload, whether timestamp/identity fields are authenticated, and whether freshness/deduplication are checked separately. Do not replace this with an unrelated hash.",
    "aggregate_save": "Inspect the shown eventstore.Save implementation and the aggregate version loaded by its caller. Verify the storage append or update rejects stale versions; then make conflict handling reload and re-evaluate the intent. A successful sequential test of this caller cannot establish concurrency safety.",
    "callback_record": "Follow RecordPartnerCallBack into the aggregate's transition rules and then into the durable save. Test both duplicate delivery and simultaneous delivery: a state guard before a save can still race unless persistence enforces the same invariant.",
    "redis_queue_client": "Follow the asynq.NewClient instance to enqueue and worker acknowledgement calls. Define the accepted-payment recovery path if Redis is unavailable; client initialization by itself provides neither a durable financial uniqueness check nor an outage policy.",
    "provider_query": "Use the observed query method as an investigation point for status recovery, retaining the same operation identity. Inspect how its result is authenticated, correlated, and applied through the ordinary guarded transition path; the query call alone does not implement a reconciler.",
    "callback_decode": "Inspect ProcessResultData before the caller records or saves its result. Establish which payload/signature bytes it verifies and how ignored, duplicate, invalid and out-of-order events are distinguished before any financial mutation.",
}


def extract_knowledge(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    knowledge = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", chunk["content"])
        useful = [
            s.strip()
            for s in sentences
            if 8 <= len(s.split()) <= 120 and ENGINEERING_WORDS.search(s) and safe_text(s)
        ]
        if not useful:
            continue
        # Extractive facts remain exact substrings of the cleaned source, with no inference mixed in.
        facts = useful[:3]
        knowledge.append(
            {
                "chunk_id": chunk["id"],
                "source_facts": facts,
                "engineering_interpretations": [
                    {
                        "scenario_id": s.key,
                        "interpretation": s.reasoning,
                        "status": "curated engineering interpretation under stated hypothetical premises; not a source quotation",
                    }
                    for s in SCENARIOS
                    if s.family == chunk["family"]
                ],
                "chunk": chunk,
            }
        )
    return knowledge


SOURCE_PREFERENCES = {
    "event_sourcing": "Event store",
    "cqrs": "Command–query separation",
    "ddd": "Domain-driven design",
    "idempotency": "Idempotence",
    "retry": "Exponential backoff",
    "callbacks": "Webhook",
    "outbox": "Inbox and outbox pattern",
    "atomicity": "Database transaction",
    "messaging": "Apache Kafka",
    "concurrency": "Optimistic concurrency control",
    "payment": "Payment gateway",
    "reconciliation": "Reconciliation (accounting)",
    "consistency": "Eventual consistency",
    "resilience": "Circuit breaker design pattern",
    "authentication": "HMAC",
    "replay": "Replay attack",
    "security": "Transport Layer Security",
    "ledger": "Double-entry bookkeeping",
    "settlement": "Settlement (finance)",
}

SCENARIO_PATTERNS = {
    "timeout_after_success": {"stable_refund_key", "provider_query"},
    "duplicate_refund": {"stable_refund_key"},
    "new_retry_identifier": {"stable_refund_key"},
    "timeout_before_response": {"stable_refund_key", "provider_query"},
    "duplicate_webhook": {"callback_record"},
    "out_of_order_webhook": {"callback_record"},
    "db_publish_gap": {"outbox_write"},
    "direct_publish": {"outbox_write", "event_publish"},
    "kafka_outage": {"event_publish"},
    "consumer_crash": {"event_publish", "outbox_write"},
    "consumer_duplicate": {"event_publish", "outbox_write"},
    "concurrent_confirmation": {"aggregate_save"},
    "cash_qr_race": {"aggregate_save"},
    "concurrent_refunds": {"aggregate_save"},
    "stale_version": {"aggregate_save"},
    "cancel_payment_race": {"callback_record", "aggregate_save"},
    "arbitrary_states": {"callback_record", "aggregate_save"},
    "missing_callback": {"provider_query"},
    "state_mismatch": {"provider_query", "aggregate_save"},
    "redis_unavailable": {"redis_queue_client"},
    "cache_only": {"redis_queue_client"},
    "unsigned_webhook": {"hmac_construction", "callback_decode"},
    "signed_replay": {"hmac_construction", "callback_decode"},
    "tls_scope": {"hmac_construction", "callback_decode"},
    "float_money": set(),
    "delete_history": set(),
    "settlement_gap": {"provider_query"},
}


def compatible(scenario: Scenario, evidence: dict[str, Any]) -> bool:
    if evidence["pattern"] not in (scenario.patterns or SCENARIO_PATTERNS.get(scenario.key, set())):
        return False
    context = (evidence["symbol"] + " " + evidence["path"]).lower()
    if scenario.key == "kafka_disabled_producer":
        return "publishfanoutes" in context
    if scenario.key in {
        "kafka_key_scope",
        "kafka_partition_expansion",
        "kafka_hot_key",
        "kafka_cross_topic_order",
    }:
        return evidence.get("call", "").endswith("ProduceRawWithKey")

    if scenario.key == "concurrent_refunds":
        return "refund" in context
    if scenario.key in {"cash_qr_race", "concurrent_confirmation"}:
        return "refund" not in context
    if scenario.key in {
        "duplicate_webhook",
        "out_of_order_webhook",
        "cancel_payment_race",
        "arbitrary_states",
    }:
        return "refund" not in context and any(
            w in context for w in ("record", "callback", "webhook", "confirm")
        )
    return True


def allocate_evidence(evidence: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group matching evidence; the splitter will join every shared file/source group."""
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        for family in item["families"]:
            result[family].append(item)
    return result


def make_record(
    scenario: Scenario, knowledge: dict[str, Any], evidence: dict[str, Any] | None
) -> dict[str, Any]:
    chunk = knowledge["chunk"]
    facts = knowledge["source_facts"]
    fact_text = " ".join(facts)
    instruction = scenario.question
    input_text = f"Hypothetical scenario (not a claim about a provider): {scenario.premise}\n\nSource facts from {chunk['title']}, section {chunk['section']}:\n{fact_text}"
    observation = ""
    bridge = ""
    if evidence:
        instruction += f" Apply the reasoning to `{evidence['symbol']}` in `{evidence['repo']}/{evidence['path']}`."
        input_text += f"\n\nLocal source evidence [{evidence['reference']}]:\n```go\n{evidence['snippet']}\n```"
        observation = f"Observed local pattern: {evidence['observation']} [{evidence['reference']}]"
        bridge = (
            f"Application to `{evidence['symbol']}`: " + PATTERN_APPLICATIONS[evidence["pattern"]]
        )
    output = f"Source fact: {facts[0]}\n\n"
    if observation:
        output += observation + "\n\n"
    output += "Engineering interpretation under the stated scenario: " + scenario.reasoning
    if bridge:
        output += "\n\n" + bridge
    output += "\n\nVerification to add: " + scenario.test
    if scenario.kind in {"state_transition", "race_condition"}:
        input_text += (
            "\n\nIllustrative state vocabulary: "
            + ", ".join(STATES)
            + ". These states are not asserted to exist in every provider or repository."
        )
    metadata = {
        "source_title": chunk["title"],
        "source_url": chunk["source_url"],
        "section": chunk["section"],
        "category": chunk["category"],
        "tags": chunk["tags"],
        "generation_type": scenario.kind,
        "source_revision": chunk["revision_id"],
        "source_page_id": chunk["page_id"],
        "chunk_id": chunk["id"],
        "family": scenario.family,
        "scenario_id": scenario.key,
        "license": chunk["license"],
        "attribution": chunk["attribution"],
        "history_url": chunk["history_url"],
        "source_facts": facts,
        "engineering_interpretation": scenario.reasoning,
        "scenario_premise": scenario.premise,
        "verification": scenario.test,
        "code_linked": evidence is not None,
        "code_evidence": [evidence] if evidence else [],
        "code_observation": evidence["observation"] if evidence else None,
        "code_application": bridge or None,
        "generator": "curated-scenario-and-observed-code-v1; no external teacher",
        "transformations": [
            "structural HTML cleaning",
            "extractive sentence selection",
            "curated hypothetical payment application",
            "optional local code observation",
        ],
    }
    return {
        "id": digest(instruction + input_text + output),
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "metadata": metadata,
    }


def generate(
    knowledge: list[dict[str, Any]], evidence: list[dict[str, Any]], per_scenario: int = 2
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= per_scenario <= 5:
        raise ValueError("Examples per scenario must be between 1 and 5")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in knowledge:
        by_family[item["chunk"]["family"]].append(item)
    local = allocate_evidence(evidence)
    rows = []
    gaps = []
    used: Counter[str] = Counter()
    for scenario in SCENARIOS:
        options = by_family[scenario.family]
        if not options:
            gaps.append({"scenario": scenario.key, "reason": "no useful fetched source facts"})
            continue
        # Prefer introductory definitions; score relevant source sentences, never invent a source.
        selected = min(
            options,
            key=lambda k: (
                k["chunk"]["title"] != SOURCE_PREFERENCES[scenario.family],
                k["chunk"]["section"]
                != {
                    "retry": "Exponential backoff algorithm",
                    "security": "Description",
                    "messaging": "Operation",
                    "cqrs": "Command Query Responsibility Segregation",
                }.get(scenario.family, "Overview"),
                -len(ENGINEERING_WORDS.findall(" ".join(k["source_facts"]))),
                k["chunk_id"],
            ),
        )
        candidates = sorted(
            (e for e in local.get(scenario.family, []) if compatible(scenario, e)),
            key=lambda e: (used[e["id"]], e["id"]),
        )[:per_scenario]
        for item in candidates:
            rows.append(make_record(scenario, selected, item))
            used[item["id"]] += 1
        rows.append(make_record(scenario, selected, None))
        if not candidates:
            gaps.append(
                {
                    "scenario": scenario.key,
                    "reason": "no allocated screened code evidence; generic candidate only",
                }
            )
    return rows, {
        "scenarios": len(SCENARIOS),
        "candidate_count": len(rows),
        "gaps": gaps,
        "evidence_allocation": {k: len(v) for k, v in sorted(local.items())},
    }
