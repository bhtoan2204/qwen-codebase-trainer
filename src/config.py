"""Environment configuration. Paths are resolved independently of the caller's cwd."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true/false or 1/0")
    return value in {"true", "1"}


@dataclass(frozen=True)
class Settings:
    workspace: Path = field(
        default_factory=lambda: Path(os.getenv("WORKSPACE_ROOT", str(PROJECT.parent))).resolve()
    )
    data: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", str(PROJECT / "data"))).resolve()
    )
    artifacts: Path = field(
        default_factory=lambda: Path(
            os.getenv("ARTIFACTS_DIR", str(PROJECT / "artifacts"))
        ).resolve()
    )
    model: str = field(default_factory=lambda: os.getenv("BASE_MODEL", "Qwen/Qwen3-8B"))
    embedding: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "hashing-v1"))
    max_file_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_FILE_BYTES", "1000000")))
    blame: bool = field(default_factory=lambda: flag("INCLUDE_BLAME"))

    def prepare(self) -> None:
        if not self.workspace.is_dir():
            raise ValueError("WORKSPACE_ROOT must be an existing directory")
        if self.max_file_bytes < 1:
            raise ValueError("MAX_FILE_BYTES must be positive")
        for path in (self.data, self.artifacts):
            path.mkdir(parents=True, exist_ok=True)
