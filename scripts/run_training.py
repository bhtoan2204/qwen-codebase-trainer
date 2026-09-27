#!/usr/bin/env python3
"""Run Axolotl with a durable local log and machine-readable lifecycle status."""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

from src.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = Path(sys.executable).parent / "axolotl"
    if not executable.is_file() or not args.config.is_file():
        raise SystemExit("Axolotl executable or config missing in this environment")
    status = {
        "state": "running",
        "supervisor_pid": os.getpid(),
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "config": str(args.config.resolve()),
        "log": str(directory / "train.log"),
    }
    environment = {
        **os.environ,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "AXOLOTL_DO_NOT_TRACK": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PATH": str(executable.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    # The host Nix shell may inject Python 3.14 packages into this Python 3.11 venv.
    environment.pop("PYTHONPATH", None)
    with (directory / "train.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(executable), "train", str(args.config.resolve())],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        status["training_pid"] = process.pid
        write_json(directory / "status.json", status)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            code = process.wait()
        status.update(
            state="completed" if code == 0 else "failed",
            exit_code=code,
            finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        write_json(directory / "status.json", status)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
