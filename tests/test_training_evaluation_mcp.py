from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from conftest import source

from src.config import Settings
from src.evaluation.runner import retrieval_evaluation, summarize_reviewed
from src.io import write_jsonl
from src.mcp.server import create_server
from src.retrieval.index import Index
from src.training.configure import render


def test_training_config_environment(workspace: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEQUENCE_LEN", "1024")
    monkeypatch.setenv("EPOCHS", "1")
    config = render(workspace, workspace.artifacts / "config.yml", validate_data=False)
    assert config["sequence_len"] == 1024
    assert config["num_epochs"] == 1
    assert config["adapter"] == "qlora"
    assert config["datasets"][0]["roles_to_train"] == ["assistant"]
    with pytest.raises(ValueError, match="Generate"):
        render(workspace, workspace.artifacts / "config.yml")


def test_mcp_tools_do_not_load_model(workspace: Settings) -> None:
    with Index(workspace.artifacts / "index") as index:
        index.update([source("package pay\nfunc Payment() {\n}\n")])
        server = create_server(workspace, index)
        tools = asyncio.run(server.list_tools())
        assert {t.name for t in tools} == {
            "search_code",
            "find_symbol",
            "get_architecture_context",
            "find_similar_implementation",
            "explain_code_context",
        }
        result = asyncio.run(server.call_tool("find_symbol", {"symbol": "Payment"}))
        assert "Payment" in str(result)


def test_metrics_do_not_invent_groundedness(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    write_jsonl(
        path,
        [
            {
                "variant": "A",
                "file_location_accuracy": 1.0,
                "latency_seconds": 1.0,
                "source_retrieval_accuracy": None,
                "review": {},
            }
        ],
    )
    report = summarize_reviewed(path, tmp_path / "summary.json")
    assert report["variants"]["A"]["answer_groundedness"] is None
    assert report["variants"]["A"]["hallucination_rate"] is None


def test_retrieval_evaluation(tmp_path: Path) -> None:
    with Index(tmp_path) as index:
        index.update([source("package pay\nfunc Payment() {\n}\n")])
        report = retrieval_evaluation(
            index,
            [{"id": "case", "question": "Payment", "expected_files": ["payments/service/pay.go"]}],
        )
        assert report["mean_recall_at_5"] == 1


def test_mcp_stdio_round_trip(workspace: Settings) -> None:
    import os
    import sys

    from mcp.client.stdio import stdio_client

    from mcp import ClientSession, StdioServerParameters

    with Index(workspace.artifacts / "index") as index:
        index.update([source("package pay\nfunc Payment() {\n}\n")])

    async def exchange() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.cli", "mcp"],
            env={
                **os.environ,
                "WORKSPACE_ROOT": str(workspace.workspace),
                "ARTIFACTS_DIR": str(workspace.artifacts),
                "DATA_DIR": str(workspace.data),
                "EMBEDDING_MODEL": "hashing-v1",
            },
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("search_code", {"query": "Payment", "top_k": 1})
                assert not result.isError
                assert "Payment" in str(result.content)

    asyncio.run(asyncio.wait_for(exchange(), timeout=20))
