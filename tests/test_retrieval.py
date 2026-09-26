from __future__ import annotations

from pathlib import Path

import pytest
from conftest import source

from src.retrieval.index import Index

PAY = "package pay\n// Idempotency prevents duplicate payment work.\nfunc Idempotency() bool {\n return true\n}\n"


def test_index_incremental_changed_deleted_and_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = source(PAY)
    with Index(tmp_path) as index:
        first = index.update([original])
        assert first["embedded_chunks"] == 2
        original_vectors = index.db.execute("SELECT id,vector FROM chunks ORDER BY id").fetchall()
        real_encode = index.embedder.encode
        calls = []

        def count(texts: list[str]) -> list[list[float]]:
            calls.append(len(texts))
            return real_encode(texts)

        monkeypatch.setattr(index.embedder, "encode", count)
        assert index.update([original])["embedded_chunks"] == 0
        assert calls == []
        changed_commit = source(PAY, commit="b" * 40)
        assert index.update([changed_commit])["embedded_chunks"] == 0
        assert calls == []
        assert all(c.commit == "b" * 40 for c in index.chunks())
        assert (
            original_vectors
            == index.db.execute("SELECT id,vector FROM chunks ORDER BY id").fetchall()
        )
        for mode in ("lexical", "vector", "hybrid"):
            hits = index.search("payment idempotency", mode=mode)
            assert hits and hits[0]["symbol"] == "Idempotency"
            assert hits[0]["reference"].startswith("payments/service/pay.go:")
        changed = source(PAY.replace("true", "false"))
        assert index.update([changed])["changed_files"] == 1
        assert index.find_symbol("Idempotency")
        assert index.update([])["deleted_files"] == 1
        assert index.search("idempotency") == []


def test_dirty_replay_preserves_vectors(tmp_path: Path) -> None:
    with Index(tmp_path) as index:
        index.update([source(PAY)])
        index.client.delete_collection("code")
        index.db.execute("INSERT INTO settings VALUES ('dirty', '1')")
        index.db.commit()
    with Index(tmp_path) as index:
        assert index.search("idempotency", mode="vector")
        assert not index.db.execute("SELECT 1 FROM settings WHERE key='dirty'").fetchone()


def test_embedding_version_change_fails(tmp_path: Path) -> None:
    with Index(tmp_path):
        pass
    with pytest.raises(ValueError, match="version changed"):
        Index(tmp_path, "different-model")


def test_no_secret_bypasses_index(tmp_path: Path) -> None:
    with Index(tmp_path) as index:
        with pytest.raises(ValueError, match="Unsafe"):
            index.update([source('package x\nconst password = "sensitive"')])


def test_security_exclusion_prunes_previously_indexed_file(tmp_path: Path) -> None:
    with Index(tmp_path) as index:
        index.update([source(PAY)])
        assert index.update([])["deleted_files"] == 1
        assert index.client.count("code").count == 0
