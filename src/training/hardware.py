from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from typing import Any


def hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "gpus": [],
        "cuda_available": False,
        "cuda_version": None,
        "pytorch_cuda_version": None,
        "bf16_supported": False,
    }
    if shutil.which("nvidia-smi"):
        full = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        match = re.search(r"CUDA Version:\s*([\d.]+)", full.stdout)
        result["cuda_version"] = match.group(1) if match else None
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if query.returncode == 0:
            result["gpus"] = [
                {
                    "model": line.rsplit(",", 1)[0].strip(),
                    "vram_gib": round(float(line.rsplit(",", 1)[1]) / 1024, 2),
                }
                for line in query.stdout.splitlines()
                if "," in line
            ]
    if importlib.util.find_spec("torch"):
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["pytorch_cuda_version"] = torch.version.cuda
        result["bf16_supported"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if torch.cuda.is_available():
            result["gpus"] = [
                {
                    "model": torch.cuda.get_device_name(i),
                    "vram_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                }
                for i in range(torch.cuda.device_count())
            ]
    else:
        result["pytorch_status"] = "not installed; CUDA usability and bf16 support unverified"
    vram = min((g["vram_gib"] for g in result["gpus"]), default=0)
    result["recommended_model"] = (
        "Qwen/Qwen3-8B"
        if vram >= 16
        else "Qwen/Qwen3-4B"
        if vram >= 10
        else "Qwen/Qwen3-1.7B"
        if vram >= 6
        else "Qwen/Qwen3-0.6B"
    )
    result["recommended_sequence_length"] = 2048 if vram >= 16 else 1024
    result["recommended_micro_batch_size"] = 1
    result["recommendation"] = (
        "8B QLoRA planning estimate: 16+ GiB, preferably 24 GiB; validate actual peak memory. Other GPU processes reduce available memory."
        if vram >= 16
        else "VRAM is below the conservative 16 GiB budget for Qwen3-8B QLoRA. Use the smaller recommended model; with no CUDA GPU, prepare retrieval/datasets only."
    )
    return result
