"""Conservative exclusion, never redaction that could change code semantics."""

from __future__ import annotations

import re
from pathlib import PurePath

VERSION = "security-v1"
BLOCKED_DIRS = {
    ".git",
    "vendor",
    "node_modules",
    "dist",
    "build",
    "generated",
    ".terraform",
    ".venv",
    "__pycache__",
    "fixtures",
    "testdata",
    "logs",
    "dumps",
}
PATH_PATTERN = re.compile(
    r"(^\.env($|\.)|credentials|secrets?|\.pem$|\.key$|\.p12$|\.pfx$|\.tfstate|id_rsa|id_ed25519|\.sqlite|\.db$|\.log$|dump|customer[_-]?data)",
    re.I,
)
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    "provider_token": re.compile(
        r"\b(?:AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})"
    ),
    "credential_literal": re.compile(
        r"""(?i)["'`]?\b(?:password|passwd|pwd|api[_-]?key|access[_-]?key|secret|token|authorization|client[_-]?secret)["'`]?\s*(?::=|[:=])\s*["'`][^"'`\n]{4,}["'`]"""
    ),
    "credential_yaml": re.compile(
        r"(?im)^\s*(?:password|passwd|api[_-]?key|secret|token|client[_-]?secret)\s*:\s*\S+"
    ),
    "credential_url": re.compile(
        r"(?i)(?:https?|postgres(?:ql)?|mysql|redis|mongodb)://[^\s/]+:[^\s/@]+@|[?&](?:token|key|secret|password|signature|credential)=[^&\s]+"
    ),
    "kubernetes_secret": re.compile(r"(?im)^\s*kind\s*:\s*[\"']?Secret[\"']?\s*$"),
    "kubernetes_secret_json": re.compile(r"""(?i)["']kind["']\s*:\s*["']Secret["']"""),
    "uuid_literal": re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"),
    "email_literal": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "opaque_literal": re.compile(r"""["'][A-Za-z0-9+/=_-]{32,}["']"""),
    "customer_literal": re.compile(
        r"""(?i)["']?(?:customer[_-]?id|account[_-]?(?:id|number)|card[_-]?number|phone|email|merchant[_-]?id)["']?\s*[:=]\s*(?:["'][^"'\n]+["']|\d{5,})"""
    ),
    "environment_value": re.compile(r"(?m)^\s*(?:export\s+)?[A-Z][A-Z0-9_]{2,}\s*=\s*\S+"),
    "sql_data": re.compile(r"(?i)\b(?:INSERT\s+INTO|COPY\s+\S+\s+FROM)\b"),
}


def excluded_path(path: str) -> bool:
    parts = PurePath(path).parts
    return any(p in BLOCKED_DIRS or PATH_PATTERN.search(p) for p in parts)


def findings(content: str) -> list[dict[str, str | int]]:
    return [
        {"rule": name, "line": content.count("\n", 0, match.start()) + 1}
        for name, pattern in PATTERNS.items()
        for match in pattern.finditer(content)
    ]


def safe(content: str) -> bool:
    return not findings(content)
