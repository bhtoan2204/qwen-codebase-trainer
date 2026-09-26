from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class SourceFile:
    repo: str
    path: str
    language: str
    commit: str
    content_hash: str
    content: str
    commit_messages: list[str] = field(default_factory=list)
    blame: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Chunk:
    repo: str
    path: str
    language: str
    symbol: str
    symbol_type: str
    start_line: int
    end_line: int
    commit: str
    content: str
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return digest(f"{self.repo}/{self.path}:{self.start_line}:{self.symbol}")[:32]

    @property
    def reference(self) -> str:
        return f"{self.repo}/{self.path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
