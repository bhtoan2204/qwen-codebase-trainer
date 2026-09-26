from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.config import Settings
from src.models import SourceFile, digest


def source(
    content: str, path: str = "service/pay.go", repo: str = "payments", commit: str = "a" * 40
) -> SourceFile:
    return SourceFile(repo, path, "go", commit, digest(content), content)


@pytest.fixture
def workspace(tmp_path: Path) -> Settings:
    repo = tmp_path / "repos" / "payments"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "pay.go").write_text(
        "package pay\n\n// Pay handles a payment.\nfunc Pay() error {\n return nil\n}\n"
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "Initial implementation",
        ],
        check=True,
    )
    return Settings(workspace=repo.parent, data=tmp_path / "data", artifacts=tmp_path / "artifacts")
