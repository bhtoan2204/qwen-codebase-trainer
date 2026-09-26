from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Settings
from src.scanner.repositories import scan
from src.security.scanner import excluded_path, findings, safe


@pytest.mark.parametrize(
    "text",
    [
        'password: "really-sensitive"',
        'api_key = "something-sensitive"',
        "-----BEGIN RSA PRIVATE KEY-----",
        "eyJabcdefghijk.abcdefghijk.abcdefghijk",
        "https://user:password@host.test/path",
        "https://host.test/?token=not-for-training",
        "kind: Secret\ndata:\n  password: abc",
        'const email = "realperson@example.test"',
        'customer_id: "customer-124"',
        "export REGION=production",
        "INSERT INTO customers VALUES (1);",
    ],
)
def test_secret_rules(text: str) -> None:
    assert not safe(text)
    report = json.dumps(findings(text))
    assert text not in report
    assert all(set(f) == {"rule", "line"} for f in findings(text))


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        "keys/server.pem",
        "credentials.json",
        "terraform.tfstate",
        "node_modules/pkg/index.js",
        "testdata/customers.json",
    ],
)
def test_path_exclusions(path: str) -> None:
    assert excluded_path(path)


def test_scan_screens_metadata_and_symlinks(workspace: Settings, tmp_path: Path) -> None:
    repo = workspace.workspace / "payments"
    (repo / "unsafe.go").write_text('package pay\nconst password = "really-sensitive"')
    (repo / ".env").write_text("TOKEN=secret")
    (repo / "generated.go").write_text("// Code generated. DO NOT EDIT.\npackage pay")
    (repo / "binary.go").write_bytes(b"\x00binary")
    external = tmp_path / "external.go"
    external.write_text("package private")
    (repo / "link.go").symlink_to(external)
    files = scan(workspace)
    assert [f.path for f in files] == ["pay.go"]
    assert len(files[0].commit) == 40
    report = (workspace.artifacts / "security-scan.json").read_text()
    assert "really-sensitive" not in report
    assert "credential_literal" in report
    assert "content" not in (workspace.artifacts / "scan-manifest.jsonl").read_text().replace(
        "content_hash", "hash"
    )


def test_blame_never_stores_identity(workspace: Settings) -> None:
    from dataclasses import replace

    files = scan(replace(workspace, blame=True))
    assert files[0].blame
    assert set(files[0].blame[0]) == {"commit", "line"}


@pytest.mark.parametrize(
    "text", ["password := `sensitive-value`", '{"kind": "Secret", "data": {"value": "YWJj"}}']
)
def test_additional_credential_encodings(text: str) -> None:
    assert not safe(text)
