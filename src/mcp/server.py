from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from src.config import Settings
from src.retrieval.index import Index


def create_server(settings: Settings, index: Index) -> FastMCP:
    server = FastMCP("qwen-codebase-trainer")

    @server.tool()
    def search_code(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Hybrid search over screened local source; no LLM calls."""
        return index.search(query, top_k)

    @server.tool()
    def find_symbol(symbol: str) -> list[dict[str, Any]]:
        """Find exact function/type names, including receiver-qualified Go methods."""
        return index.find_symbol(symbol)

    @server.tool()
    def get_architecture_context(query: str) -> list[dict[str, Any]]:
        """Retrieve architecture evidence without generating architectural claims."""
        return index.search(query + " architecture package interface service dependencies", 8)

    @server.tool()
    def find_similar_implementation(query: str) -> list[dict[str, Any]]:
        """Retrieve implementation candidates; similarity does not imply equivalence."""
        return index.search(query, 8)

    @server.tool()
    def explain_code_context(query: str, summarize: bool = False) -> dict[str, Any]:
        """Return source context; explicitly request summarize=true to load local Qwen."""
        context = index.search(query, 5)
        result: dict[str, Any] = {"sources": context}
        if summarize:
            import os

            from src.inference.qwen import Qwen

            model = Qwen(settings.model, os.getenv("ADAPTER_PATH") or None)
            result["answer"] = model.answer(query, context)
        return result

    return server


def serve(settings: Settings) -> None:
    with Index(settings.artifacts / "index", settings.embedding) as index:
        create_server(settings, index).run(transport="stdio")
