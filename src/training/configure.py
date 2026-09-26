from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT, Settings, flag
from src.io import read_jsonl, write_text
from src.security.scanner import safe
from src.training.hardware import hardware


def render(
    settings: Settings, output: Path, overrides: Path | None = None, validate_data: bool = True
) -> dict[str, Any]:
    config = yaml.safe_load((PROJECT / "configs/qwen3-8b-qlora.yml").read_text())
    config["base_model"] = settings.model
    config["output_dir"] = str(
        Path(os.getenv("OUTPUT_DIR", str(PROJECT / "outputs/qwen3-8b-lora"))).resolve()
    )
    config["dataset_prepared_path"] = str(settings.artifacts / "axolotl-prepared")
    config["datasets"][0]["path"] = str(settings.data / "train.jsonl")
    config["test_datasets"][0]["path"] = str(settings.data / "validation.jsonl")
    for env, key, converter in [
        ("SEQUENCE_LEN", "sequence_len", int),
        ("EPOCHS", "num_epochs", float),
        ("MICRO_BATCH_SIZE", "micro_batch_size", int),
        ("GRADIENT_ACCUMULATION_STEPS", "gradient_accumulation_steps", int),
        ("LORA_R", "lora_r", int),
        ("LORA_ALPHA", "lora_alpha", int),
        ("LEARNING_RATE", "learning_rate", float),
    ]:
        if env in os.environ:
            value = converter(os.environ[env])
            if value <= 0:
                raise ValueError(f"{env} must be positive")
            config[key] = value
    config["flash_attention"] = flag("FLASH_ATTENTION")
    gpu = hardware()
    if gpu["cuda_available"]:
        config["bf16"] = gpu["bf16_supported"]
        config["fp16"] = not gpu["bf16_supported"]
    if overrides:
        values = yaml.safe_load(overrides.read_text())
        if not isinstance(values, dict):
            raise ValueError("Training overrides must be a YAML mapping")
        config.update(values)
    if config.get("adapter") != "qlora" or not config.get("load_in_4bit"):
        raise ValueError("This training pipeline requires adapter=qlora and load_in_4bit=true")
    if validate_data:
        for key in ("datasets", "test_datasets"):
            for dataset in config[key]:
                path = Path(dataset["path"])
                if not path.is_file():
                    raise ValueError(
                        "Generate and inspect datasets before preparing training config"
                    )
                rows = read_jsonl(path)
                if not rows:
                    raise ValueError(
                        f"{path.name} is empty; scan more modules rather than leaking samples between splits"
                    )
                for row in rows:
                    messages = row.get("messages", [])
                    if (
                        len(messages) < 2
                        or messages[-1].get("role") != "assistant"
                        or not safe(str(messages))
                    ):
                        raise ValueError("Training dataset failed Messages/security validation")
    write_text(output, yaml.safe_dump(config, sort_keys=False))
    return config
