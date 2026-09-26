from __future__ import annotations

import re
from typing import Protocol

from src.models import Chunk, SourceFile

VERSION = "parser-v2"


class CodeParser(Protocol):
    def parse(self, source: SourceFile) -> list[Chunk]: ...


def chunk(
    source: SourceFile, start: int, end: int, symbol: str, kind: str, facts: dict | None = None
) -> Chunk:
    return Chunk(
        source.repo,
        source.path,
        source.language,
        symbol,
        kind,
        start,
        end,
        source.commit,
        "\n".join(source.content.splitlines()[start - 1 : end]),
        facts or {},
    )


class GoParser:
    """AST top-level declarations, with attached doc comments and structural evidence."""

    def parse(self, source: SourceFile) -> list[Chunk]:
        import tree_sitter_go
        from tree_sitter import Language, Parser

        parser = Parser(Language(tree_sitter_go.language()))
        tree = parser.parse((source.content.rstrip("\n") + "\n").encode())
        if tree.root_node.has_error:
            raise ValueError(f"Invalid Go syntax in {source.repo}/{source.path}")
        output: list[Chunk] = []
        pending = []
        imports = []
        package = ""
        for node in tree.root_node.named_children:
            text = (node.text or b"").decode()
            if node.type == "import_declaration":
                imports.extend(re.findall(r'"([^"\n]+)"', text))
            if node.type == "package_clause":
                package = text.removeprefix("package").strip()
        for node in tree.root_node.named_children:
            if node.type == "comment":
                pending.append(node)
                continue
            name_node = node.child_by_field_name("name")
            kind = node.type.removesuffix("_declaration").removesuffix("_clause")
            symbol = (name_node.text or b"").decode() if name_node else kind
            if node.type in {"const_declaration", "var_declaration"}:
                names = [
                    (name.text or b"").decode()
                    for spec in node.named_children
                    for name in spec.children_by_field_name("name")
                ]
                symbol = ", ".join(names) or symbol
            if node.type == "type_declaration":
                specs = [n for n in node.named_children if n.type in {"type_spec", "type_alias"}]
                names = []
                for spec in specs:
                    n = spec.child_by_field_name("name")
                    if n:
                        names.append((n.text or b"").decode())
                    t = spec.child_by_field_name("type")
                    if len(specs) == 1 and t and t.type in {"struct_type", "interface_type"}:
                        kind = t.type.removesuffix("_type")
                symbol = ", ".join(names) or symbol
            if node.type == "method_declaration":
                receiver = node.child_by_field_name("receiver")
                if receiver:
                    symbol = (receiver.text or b"").decode() + "." + symbol
            if node.type == "package_clause":
                symbol = package
            start = node.start_point.row + 1
            doc = ""
            if pending and pending[-1].end_point.row + 1 >= node.start_point.row:
                start = pending[0].start_point.row + 1
                doc = "\n".join((n.text or b"").decode() for n in pending)
            pending = []
            calls: set[str] = set()
            constructs: set[str] = set()
            stack = [node]
            while stack:
                child = stack.pop()
                if child.type == "call_expression":
                    function = child.child_by_field_name("function")
                    if function:
                        calls.add((function.text or b"").decode())
                if child.type in {
                    "go_statement",
                    "defer_statement",
                    "select_statement",
                    "send_statement",
                    "receive_statement",
                    "if_statement",
                    "return_statement",
                }:
                    constructs.add(child.type)
                stack.extend(child.named_children)
            facts = {
                "package": package,
                "imports": imports,
                "calls": sorted(calls),
                "constructs": sorted(constructs),
                "doc": doc,
            }
            output.append(chunk(source, start, node.end_point.row + 1, symbol, kind, facts))
        return output


class StructuredParser:
    """Section/statement boundaries for text formats; line windows only for oversized units."""

    def parse(self, source: SourceFile) -> list[Chunk]:
        lines = source.content.splitlines()
        if not lines:
            return []
        boundary = {
            "markdown": r"^#{1,6}\s+",
            "sql": r"(?i)^\s*(CREATE|ALTER|COMMENT|BEGIN|COMMIT)\b",
            "protobuf": r"^\s*(message|service|enum|rpc|package|import)\b",
            "yaml": r"^(---|[A-Za-z_][\w.-]*\s*:)",
            "json": r'^\s{0,2}"[^"\n]+"\s*:',
            "shell": r"^(?:function\s+\w+|\w+\s*\(\s*\))",
        }.get(source.language, r"^$")
        starts = sorted({0, *(i for i, line in enumerate(lines) if re.search(boundary, line))})
        output = []
        for begin, end in zip(starts, starts[1:] + [len(lines)], strict=True):
            for position in range(begin, end, 160):
                finish = min(position + 160, end)
                if any(line.strip() for line in lines[position:finish]):
                    label = lines[begin].strip()[:100] or f"section-{begin + 1}"
                    output.append(chunk(source, position + 1, finish, label, "section"))
        return output


def parse(source: SourceFile) -> list[Chunk]:
    implementation: CodeParser = GoParser() if source.language == "go" else StructuredParser()
    return implementation.parse(source)
