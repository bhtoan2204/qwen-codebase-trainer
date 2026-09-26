from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np


def tokens(text: str) -> list[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z][a-z0-9]*", expanded.lower())


class Embedder:
    """Hashing is a zero-download baseline, not a semantic model."""

    def __init__(self, model: str) -> None:
        self.name = model
        self.model: Any = None
        self.dimension = 384
        if model != "hashing-v1":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install .[embeddings] for semantic embedding models") from exc
            self.model = SentenceTransformer(model, trust_remote_code=False)
            self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.model is not None:
            return self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()  # type: ignore[no-any-return]
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokens(text):
                h = hashlib.sha256(token.encode()).digest()
                vectors[i, int.from_bytes(h[:4], "big") % self.dimension] += 1 if h[4] % 2 else -1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.maximum(norms, 1e-12)).tolist()
