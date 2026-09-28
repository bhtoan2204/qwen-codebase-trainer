# CPU image: scan/index/dataset pipeline plus LoRA training with PyTorch CPU.
FROM docker.io/library/python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.6" \
    && pip install --no-cache-dir "transformers>=4.51,<5" "peft>=0.15,<1" "accelerate>=1.5" datasets

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir -e .

ENV HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "src.cli"]
