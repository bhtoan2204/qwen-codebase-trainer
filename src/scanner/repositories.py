from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.config import PROJECT, Settings
from src.io import write_json, write_jsonl
from src.models import SourceFile, digest
from src.security.scanner import excluded_path, findings, safe

LOG = logging.getLogger(__name__)
LANGUAGES = {
    ".go": "go",
    ".sql": "sql",
    ".proto": "protobuf",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".sh": "shell",
    ".bash": "shell",
}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"Git operation {args[0]} failed in repository {repo.name}")
    return result.stdout.decode("utf-8", errors="replace")


def discover(settings: Settings) -> list[Path]:
    return sorted(
        p
        for p in settings.workspace.iterdir()
        if p.is_dir() and not p.is_symlink() and p.resolve() != PROJECT and (p / ".git").exists()
    )


def scan(settings: Settings, limit: int | None = None) -> list[SourceFile]:
    settings.prepare()
    files: list[SourceFile] = []
    report: list[dict[str, object]] = []
    repos = discover(settings)
    for repo in repos:
        commit = git(repo, "rev-parse", "HEAD").strip()
        messages = [m for m in git(repo, "log", "-5", "--format=%s").splitlines() if safe(m)]
        paths = sorted(
            set(
                git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(
                    "\0"
                )
            )
        )
        for relative in paths:
            if not relative:
                continue
            path = repo / relative
            if excluded_path(relative):
                report.append({"repo": repo.name, "path": relative, "rule": "excluded_path"})
                continue
            if path.suffix not in LANGUAGES or not path.is_file():
                continue
            if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != repo.parent):
                report.append({"repo": repo.name, "path": relative, "rule": "symlink"})
                continue
            if path.stat().st_size > settings.max_file_bytes:
                report.append({"repo": repo.name, "path": relative, "rule": "oversized"})
                continue
            raw = path.read_bytes()
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if "\0" in content:
                continue
            if "DO NOT EDIT" in content[:4096] or ".pb.go" in relative or ".gen." in relative:
                continue
            issues = findings(content)
            if issues:
                report.extend({"repo": repo.name, "path": relative, **issue} for issue in issues)
                continue
            blame: list[dict[str, object]] = []
            if settings.blame:
                result = subprocess.run(
                    ["git", "-C", str(repo), "blame", "--line-porcelain", "--", relative],
                    capture_output=True,
                    timeout=30,
                )
                # Only commit/line metadata; never persist author names, emails or source lines.
                for line in result.stdout.decode(errors="replace").splitlines():
                    parts = line.split()
                    if (
                        len(parts) in (3, 4)
                        and len(parts[0]) == 40
                        and all(c in "0123456789abcdef" for c in parts[0])
                        and parts[1].isdigit()
                        and parts[2].isdigit()
                    ):
                        blame.append({"commit": parts[0], "line": int(parts[2])})
            files.append(
                SourceFile(
                    repo.name,
                    relative,
                    LANGUAGES[path.suffix],
                    commit,
                    digest(content),
                    content,
                    messages,
                    blame,
                )
            )
    write_json(
        settings.artifacts / "security-scan.json",
        {
            "repositories": len(repos),
            "accepted_files": len(files),
            "findings": report,
            "policy": "whole-file exclusion; findings contain no matched values",
        },
    )
    # Scan the entire workspace even when downstream validation is limited.
    selected = files[:limit] if limit is not None else files
    write_jsonl(
        settings.artifacts / "scan-manifest.jsonl",
        (
            {
                "repo": f.repo,
                "path": f.path,
                "commit": f.commit,
                "content_hash": f.content_hash,
                "commit_messages": f.commit_messages,
                "blame": f.blame,
            }
            for f in selected
        ),
    )
    LOG.info(
        "Scanned %d repositories; accepted %d files; selected %d",
        len(repos),
        len(files),
        len(selected),
    )
    return selected
