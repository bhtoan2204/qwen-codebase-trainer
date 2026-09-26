from __future__ import annotations

from pathlib import Path

import pytest
from conftest import source

from src.dataset.generator import assignments, generate
from src.dataset.teacher import CompatibleTeacher, DisabledTeacher, candidates, import_reviewed
from src.io import read_jsonl, write_jsonl
from src.parser.chunks import parse

CODE = """package pay
import "fmt"
// Service coordinates payment work.
type Service struct { Count int }
type Handler interface { Pay() error }
// Pay validates work before returning.
func (s *Service) Pay() error {
    text := "a brace } inside a string"
    _ = text
    return fmt.Errorf("not implemented")
}
"""


def test_go_semantics_and_ranges() -> None:
    chunks = parse(source(CODE))
    assert {c.symbol_type for c in chunks} >= {"package", "import", "struct", "interface", "method"}
    method = next(c for c in chunks if c.symbol_type == "method")
    assert method.symbol == "(s *Service).Pay"
    assert method.start_line == 6 and method.end_line == 11
    assert method.facts["calls"] == ["fmt.Errorf"]
    assert method.content.startswith("// Pay")
    assert method.content.endswith("}")


def test_invalid_go_fails() -> None:
    with pytest.raises(ValueError, match="Invalid Go"):
        parse(source("package x\nfunc Broken( {"))


def test_dataset_deterministic_and_no_file_leakage(tmp_path: Path) -> None:
    files = [source(CODE.replace("Service", f"Service{i}"), f"module{i}/pay.go") for i in range(30)]
    generate(files, tmp_path)
    before = (tmp_path / "train.jsonl").read_bytes()
    generate(files, tmp_path)
    assert before == (tmp_path / "train.jsonl").read_bytes()
    provenance = read_jsonl(tmp_path / "provenance.jsonl")
    sets = {
        split: {r["source"]["path"] for r in provenance if r["split"] == split}
        for split in ("train", "validation")
    }
    assert sets["train"] and sets["validation"]
    assert not sets["train"] & sets["validation"]
    assert all(set(row) == {"messages"} for row in read_jsonl(tmp_path / "train.jsonl"))


def test_duplicate_declarations_cannot_cross_split() -> None:
    files = [source(CODE, "a/pay.go", "first"), source(CODE, "b/pay.go", "second")]
    splits = assignments(files, [c for f in files for c in parse(f)], 0.5)
    assert len(set(splits.values())) == 1


def test_teacher_requires_external_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEACHER_BASE_URL", "https://teacher.example.test/v1")
    monkeypatch.delenv("ALLOW_EXTERNAL_TEACHER", raising=False)
    with pytest.raises(ValueError, match="ALLOW_EXTERNAL_TEACHER"):
        CompatibleTeacher()
    with pytest.raises(ValueError, match="disabled"):
        DisabledTeacher().complete("code")


def test_teacher_review_gate(tmp_path: Path) -> None:
    chunks = parse(source(CODE))
    generate([source(CODE)], tmp_path)

    class FakeTeacher:
        def complete(self, prompt: str) -> str:
            return (
                "This evidence supports only local observations; additional context is needed. "
                + chunks[0].reference
            )

    # Use one chunk so the returned reference matches its evidence.
    candidates(chunks[:1], FakeTeacher(), tmp_path / "candidates.jsonl", 1)
    assert import_reviewed(tmp_path / "candidates.jsonl", tmp_path, chunks) == 0


def test_reviewed_stale_source_rejected(tmp_path: Path) -> None:
    generate([source(CODE)], tmp_path)
    row = read_jsonl(tmp_path / "provenance.jsonl")[0]
    row.update(approved=True, reviewer="maintainer")
    row["source"]["content_hash"] = "stale"
    write_jsonl(tmp_path / "review.jsonl", [row])
    with pytest.raises(ValueError, match="stale"):
        import_reviewed(tmp_path / "review.jsonl", tmp_path, parse(source(CODE)))


def test_structured_markdown_sections() -> None:
    from dataclasses import replace

    chunks = parse(replace(source("# One\nBody\n# Two\nOther\n"), language="markdown"))
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 2), (3, 4)]


def test_go_without_final_newline() -> None:
    assert parse(source("package pay\ntype Payment struct { Amount int }"))[-1].end_line == 2


def test_teacher_candidates_cover_tasks_and_stay_unapproved(tmp_path: Path) -> None:
    chunks = [c for c in parse(source(CODE)) if c.symbol_type == "method"]

    class FakeTeacher:
        def complete(self, prompt: str) -> str:
            return (
                "The following explanation is a candidate requiring review against source evidence: "
                + chunks[0].reference
            )

    count = candidates(chunks, FakeTeacher(), tmp_path / "candidates.jsonl", 14)
    rows = read_jsonl(tmp_path / "candidates.jsonl")
    assert count == 14
    assert len({row["task"] for row in rows}) == 14
    assert all(row["approved"] is False for row in rows)


def test_go_constants_have_declared_names() -> None:
    chunks = parse(source("package pay\nconst (\n Pending = 1\n Complete = 2\n)\n"))
    assert chunks[-1].symbol == "Pending, Complete"
