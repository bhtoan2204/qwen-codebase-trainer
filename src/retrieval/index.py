from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from filelock import FileLock
from qdrant_client import QdrantClient, models

from src.models import Chunk, SourceFile
from src.parser.chunks import VERSION as PARSER_VERSION
from src.parser.chunks import parse
from src.retrieval.embeddings import Embedder, tokens
from src.security.scanner import VERSION as SECURITY_VERSION
from src.security.scanner import safe


class Index(AbstractContextManager["Index"]):
    """SQLite authoritative metadata + durable vectors; Qdrant local search projection.

    A dirty marker allows interrupted projection writes to be replayed without embedding again.
    One process owns the index at a time (including MCP). No network service is required.
    """

    def __init__(self, directory: Path, embedding: str = "hashing-v1") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(directory / "index.lock"), timeout=10)
        self.lock.acquire()
        try:
            self.db = sqlite3.connect(directory / "metadata.sqlite3")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS files(key TEXT PRIMARY KEY, hash TEXT, commit_hash TEXT);
                CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, file_key TEXT, payload TEXT, vector TEXT);
            """)
            version = f"{embedding}/{PARSER_VERSION}/{SECURITY_VERSION}"
            old = self.db.execute("SELECT value FROM settings WHERE key='version'").fetchone()
            if old and old[0] != version:
                raise ValueError(
                    "Index model/parser/security version changed. Use a new ARTIFACTS_DIR or remove the old index explicitly."
                )
            self.embedder = Embedder(embedding)
            self.client = QdrantClient(path=str(directory / "qdrant"))
            self.db.execute("INSERT OR IGNORE INTO settings VALUES ('version', ?)", (version,))
            self.db.commit()
            if not self.client.collection_exists("code"):
                self.client.create_collection(
                    "code",
                    vectors_config=models.VectorParams(
                        size=self.embedder.dimension, distance=models.Distance.COSINE
                    ),
                )
                self._replay()
            elif self.db.execute("SELECT 1 FROM settings WHERE key='dirty'").fetchone():
                self._replay()
        except Exception:
            if hasattr(self, "client"):
                self.client.close()
            if hasattr(self, "db"):
                self.db.close()
            self.lock.release()
            raise

    def _replay(self) -> None:
        self.client.delete_collection("code")
        self.client.create_collection(
            "code",
            vectors_config=models.VectorParams(
                size=self.embedder.dimension, distance=models.Distance.COSINE
            ),
        )
        rows = self.db.execute("SELECT id, vector FROM chunks").fetchall()
        for offset in range(0, len(rows), 128):
            self.client.upsert(
                "code",
                [
                    models.PointStruct(id=id_, vector=json.loads(vector))
                    for id_, vector in rows[offset : offset + 128]
                ],
            )
        self.db.execute("DELETE FROM settings WHERE key='dirty'")
        self.db.commit()

    def update(self, files: list[SourceFile], *, prune: bool = True) -> dict[str, int]:
        stats = {"changed_files": 0, "unchanged_files": 0, "deleted_files": 0, "embedded_chunks": 0}
        incoming = {f"{f.repo}/{f.path}" for f in files}
        existing = {row[0] for row in self.db.execute("SELECT key FROM files")}
        removals = existing - incoming if prune else set()
        for source in files:
            if not safe(source.content):
                raise ValueError("Unsafe source reached index; run a fresh security scan")
            key = f"{source.repo}/{source.path}"
            old = self.db.execute(
                "SELECT hash, commit_hash FROM files WHERE key=?", (key,)
            ).fetchone()
            if old and old[0] == source.content_hash:
                stats["unchanged_files"] += 1
                if old[1] != source.commit:
                    for id_, payload in self.db.execute(
                        "SELECT id,payload FROM chunks WHERE file_key=?", (key,)
                    ).fetchall():
                        data = json.loads(payload)
                        data["commit"] = source.commit
                        self.db.execute(
                            "UPDATE chunks SET payload=? WHERE id=?", (json.dumps(data), id_)
                        )
                    self.db.execute(
                        "UPDATE files SET commit_hash=? WHERE key=?", (source.commit, key)
                    )
                    self.db.commit()
                continue
            chunks = parse(source)
            vectors = self.embedder.encode(
                [f"{c.repo}/{c.path} {c.symbol}\n{c.content}" for c in chunks]
            )
            old_ids = [
                row[0] for row in self.db.execute("SELECT id FROM chunks WHERE file_key=?", (key,))
            ]
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO settings VALUES ('dirty', '1')")
                self.db.execute("DELETE FROM chunks WHERE file_key=?", (key,))
                self.db.execute(
                    "INSERT OR REPLACE INTO files VALUES (?, ?, ?)",
                    (key, source.content_hash, source.commit),
                )
                self.db.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                    [
                        (c.id, key, json.dumps(asdict(c)), json.dumps(v))
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
            if old_ids:
                self.client.delete("code", models.PointIdsList(points=old_ids))
            if chunks:
                self.client.upsert(
                    "code",
                    [
                        models.PointStruct(id=c.id, vector=v)
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
            self.db.execute("DELETE FROM settings WHERE key='dirty'")
            self.db.commit()
            stats["changed_files"] += 1
            stats["embedded_chunks"] += len(chunks)
        for key in removals:
            ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE file_key=?", (key,))]
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO settings VALUES ('dirty', '1')")
                self.db.execute("DELETE FROM chunks WHERE file_key=?", (key,))
                self.db.execute("DELETE FROM files WHERE key=?", (key,))
            if ids:
                self.client.delete("code", models.PointIdsList(points=ids))
            self.db.execute("DELETE FROM settings WHERE key='dirty'")
            self.db.commit()
            stats["deleted_files"] += 1
        return stats

    def chunks(self) -> list[Chunk]:
        return [
            Chunk(**json.loads(row[0]))
            for row in self.db.execute("SELECT payload FROM chunks ORDER BY file_key,id")
        ]

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict[str, Any]]:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if mode not in {"lexical", "vector", "hybrid"}:
            raise ValueError("mode must be lexical, vector or hybrid")
        if not query.strip():
            return []
        # Package/import-only chunks have tiny bodies and otherwise outrank real
        # implementations merely because their file paths match the question.
        chunks = [c for c in self.chunks() if c.symbol_type not in {"package", "import"}]
        if not chunks:
            return []
        searchable = {c.id for c in chunks}
        scores: dict[str, float] = {}
        rankings: list[list[tuple[str, float]]] = []
        if mode in {"lexical", "hybrid"}:
            docs = [Counter(tokens(f"{c.repo}/{c.path} {c.symbol} {c.content}")) for c in chunks]
            average = sum(sum(d.values()) for d in docs) / len(docs) or 1
            terms = set(tokens(query))
            frequency = {term: sum(term in d for d in docs) for term in terms}
            lexical = []
            for c, d in zip(chunks, docs, strict=True):
                score = sum(
                    math.log(1 + (len(docs) - frequency[t] + 0.5) / (frequency[t] + 0.5))
                    * d[t]
                    * 2.5
                    / (d[t] + 1.5 * (0.25 + 0.75 * sum(d.values()) / average))
                    for t in terms
                    if d[t]
                )
                if score > 0:
                    lexical.append((c.id, score))
            rankings.append(sorted(lexical, key=lambda p: (-p[1], p[0]))[: max(50, top_k * 4)])
        if mode in {"vector", "hybrid"}:
            result = self.client.query_points(
                "code", query=self.embedder.encode([query])[0], limit=max(50, top_k * 4)
            )
            rankings.append(
                [
                    (str(point.id).replace("-", ""), point.score)
                    for point in result.points
                    if point.score > 0 and str(point.id).replace("-", "") in searchable
                ]
            )
        for ranking in rankings:
            for rank, (id_, score) in enumerate(ranking, 1):
                scores[id_] = scores.get(id_, 0) + (1 / (60 + rank) if mode == "hybrid" else score)
        lookup = {c.id: c for c in chunks}
        return [
            {**lookup[id_].to_dict(), "score": score, "reference": lookup[id_].reference}
            for id_, score in sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:top_k]
        ]

    def find_symbol(self, symbol: str) -> list[dict[str, Any]]:
        return [
            {**c.to_dict(), "reference": c.reference}
            for c in self.chunks()
            if c.symbol == symbol or c.symbol.endswith("." + symbol)
        ][:100]

    def __exit__(self, *args: object) -> None:
        self.client.close()
        self.db.close()
        self.lock.release()
