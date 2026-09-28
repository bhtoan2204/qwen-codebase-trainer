#!/usr/bin/env python3
"""LoRA fine-tuning on CPU (no CUDA/bitsandbytes/Axolotl) with assistant-only loss."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

DATA = Path(os.getenv("DATA_DIR", "data"))
MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
SEQ = int(os.getenv("SEQUENCE_LEN", "1024"))
OUTPUT = Path(os.getenv("OUTPUT_DIR", "outputs/qwen3-0.6b-lora-cpu"))
# Location tasks only restate the declaration; the earlier GPU run excluded them too.
SKIP_TASKS = set(filter(None, os.getenv("SKIP_TASKS", "location").split(",")))


def skipped_messages() -> set[str]:
    """Train/validation rows carry only messages; the task label lives in provenance."""
    skipped = set()
    for line in (DATA / "provenance.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["task"] in SKIP_TASKS:
            skipped.add(json.dumps(row["messages"], sort_keys=True))
    return skipped


def load(name: str, tokenizer: Any, skipped: set[str]) -> list[dict[str, list[int]]]:
    rows, dropped = [], 0
    for line in (DATA / name).read_text(encoding="utf-8").splitlines():
        messages = json.loads(line)["messages"]
        if json.dumps(messages, sort_keys=True) in skipped:
            continue
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer = tokenizer(messages[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)[
            "input_ids"
        ]
        if len(prefix) + len(answer) > SEQ:
            dropped += 1
            continue
        rows.append(
            {
                "input_ids": prefix + answer,
                "attention_mask": [1] * (len(prefix) + len(answer)),
                "labels": [-100] * len(prefix) + answer,
            }
        )
    print(f"{name}: {len(rows)} examples, {dropped} dropped (> {SEQ} tokens)", flush=True)
    return rows


def main() -> None:
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", str(os.cpu_count() or 4))))
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, "
          f"cuda {torch.cuda.is_available()}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    skipped = skipped_messages()
    train = load("train.jsonl", tokenizer, skipped)
    validation = load("validation.jsonl", tokenizer, skipped)
    if not train:
        raise SystemExit("empty training set; run `dataset` first")

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16 if use_cuda else torch.float32
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(os.getenv("LORA_R", "16")),
            lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT),
        num_train_epochs=float(os.getenv("EPOCHS", "1")),
        max_steps=int(os.getenv("MAX_STEPS", "-1")),
        per_device_train_batch_size=int(os.getenv("MICRO_BATCH_SIZE", "2")),
        per_device_eval_batch_size=int(os.getenv("MICRO_BATCH_SIZE", "2")),
        gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8")),
        learning_rate=float(os.getenv("LEARNING_RATE", "0.0002")),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=1,
        eval_strategy="steps" if validation else "no",
        eval_steps=int(os.getenv("EVAL_STEPS", "25")),
        save_strategy="steps",
        save_steps=int(os.getenv("SAVE_STEPS", "25")),
        save_total_limit=2,
        group_by_length=True,
        bf16=use_cuda,
        use_cpu=not use_cuda,
        dataloader_num_workers=0,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=validation or None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    started = time.time()
    last = OUTPUT.is_dir() and any(OUTPUT.glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if last else None)
    trainer.save_model(str(OUTPUT))
    tokenizer.save_pretrained(str(OUTPUT))
    metrics = {**result.metrics, "seconds": round(time.time() - started, 1), "base_model": MODEL,
               "train_examples": len(train), "validation_examples": len(validation)}
    if validation:
        metrics.update(trainer.evaluate())
    (OUTPUT / "result.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
