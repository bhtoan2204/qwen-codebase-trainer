from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from src.config import PROJECT, Settings
from src.io import read_jsonl, write_json


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Offline code retrieval and Qwen training preparation"
    )
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("scan", "index", "dataset"):
        command = sub.add_parser(name)
        command.add_argument(
            "--limit",
            type=int,
            help="Bound files for a sample; index samples use a separate directory",
        )
        if name == "dataset":
            from src.dataset.wikipedia.pipeline import add_cli

            add_cli(command)
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    chat = sub.add_parser("chat")
    chat.add_argument("--no-rag", action="store_true")
    chat.add_argument("--base", action="store_true", help="Ignore ADAPTER_PATH")
    chat.add_argument("--question")
    sub.add_parser("mcp")
    sub.add_parser("gpu")
    config = sub.add_parser("training-config")
    config.add_argument("--output", type=Path, default=PROJECT / "configs/generated.yml")
    config.add_argument("--overrides", type=Path)
    teach = sub.add_parser("teacher")
    teach.add_argument("--limit", type=int, default=14, help="Maximum teacher calls")
    review = sub.add_parser("import-reviewed")
    review.add_argument("path", type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--compare", action="store_true")
    evaluate.add_argument("--reviewed", type=Path)
    evaluate.add_argument("--questions", type=Path)
    return root


def run(args: argparse.Namespace, settings: Settings) -> None:
    if args.command == "dataset" and getattr(args, "dataset_kind", None) == "wikipedia":
        from src.dataset.wikipedia.pipeline import run as run_wikipedia

        print(json.dumps(run_wikipedia(args, settings), indent=2))
        return

    from src.retrieval.index import Index
    from src.scanner.repositories import scan

    if args.command in {"scan", "index", "dataset"}:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        files = scan(settings, args.limit)
        if args.command == "scan":
            print(
                json.dumps(
                    {"files": len(files), "report": str(settings.artifacts / "security-scan.json")}
                )
            )
        elif args.command == "dataset":
            from src.dataset.generator import generate

            print(json.dumps(generate(files, settings.data)))
        else:
            directory = settings.artifacts / ("sample-index" if args.limit else "index")
            with Index(directory, settings.embedding) as index:
                print(json.dumps({**index.update(files), "index": str(directory)}))
    elif args.command == "search":
        with Index(settings.artifacts / "index", settings.embedding) as index:
            print(json.dumps(index.search(args.query, args.top_k, args.mode), indent=2))
    elif args.command == "gpu":
        from src.training.hardware import hardware

        print(json.dumps(hardware(), indent=2))
    elif args.command == "training-config":
        from src.training.configure import render

        render(settings, args.output, args.overrides)
        print(f"Wrote {args.output}")
    elif args.command == "chat":
        from src.inference.qwen import Qwen

        model = Qwen(settings.model, None if args.base else os.getenv("ADAPTER_PATH") or None)
        with Index(settings.artifacts / "index", settings.embedding) as index:
            while True:
                try:
                    question = args.question or input("You> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question in {"exit", "quit"}:
                    break
                if not question:
                    continue
                context = [] if args.no_rag else index.search(question)
                print(model.answer(question, context))
                if args.question:
                    break
    elif args.command == "mcp":
        from src.mcp.server import serve

        serve(settings)
    elif args.command in {"teacher", "import-reviewed"}:
        from src.dataset.teacher import candidates, import_reviewed, teacher
        from src.parser.chunks import parse

        files = scan(settings)
        chunks = [c for f in files for c in parse(f)]
        if args.command == "teacher":
            count = candidates(
                chunks, teacher(settings), settings.data / "teacher-candidates.jsonl", args.limit
            )
        else:
            count = import_reviewed(args.path, settings.data, chunks)
        print(json.dumps({"examples": count}))
    elif args.command == "evaluate":
        from src.evaluation.runner import (
            compare,
            questions,
            retrieval_evaluation,
            summarize_reviewed,
        )

        if args.reviewed:
            print(
                json.dumps(
                    summarize_reviewed(
                        args.reviewed, settings.artifacts / "evaluation-summary.json"
                    ),
                    indent=2,
                )
            )
            return
        path = args.questions or settings.data / "eval-questions.jsonl"
        if not args.questions:
            questions(settings.data / "provenance.jsonl", path)
        cases = read_jsonl(path)
        with Index(settings.artifacts / "index", settings.embedding) as index:
            if args.compare:
                adapter = os.getenv("ADAPTER_PATH")
                if not adapter:
                    raise ValueError(
                        "Four-way comparison requires ADAPTER_PATH from completed training"
                    )
                compare(
                    settings.model,
                    adapter,
                    index,
                    cases,
                    settings.artifacts / "evaluation-responses.jsonl",
                )
            else:
                report = retrieval_evaluation(index, cases)
                write_json(settings.artifacts / "retrieval-evaluation.json", report)
                print(
                    json.dumps(
                        {"mean_recall_at_5": report["mean_recall_at_5"], "cases": len(cases)}
                    )
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parser().parse_args()
    try:
        settings = Settings()
        settings.prepare()
        run(args, settings)
    except (ValueError, RuntimeError, OSError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
