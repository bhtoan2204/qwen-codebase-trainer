PYTHON ?= python
AXOLOTL ?= axolotl
CONFIG ?= configs/generated.yml
ADAPTER_PATH ?= outputs/qwen3-8b-lora

.PHONY: check scan index dataset config preprocess train merge infer mcp gpu
check:
	$(PYTHON) -m pytest
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy src scripts
scan:
	$(PYTHON) -m src.cli scan
index:
	$(PYTHON) -m src.cli index
dataset:
	$(PYTHON) -m src.cli dataset
config:
	$(PYTHON) -m src.cli training-config --output $(CONFIG)
preprocess: config
	$(AXOLOTL) preprocess $(CONFIG) --debug
train: config
	$(AXOLOTL) train $(CONFIG)
merge:
	$(AXOLOTL) merge-lora $(CONFIG) --lora-model-dir="$(ADAPTER_PATH)"
infer:
	ADAPTER_PATH="$(ADAPTER_PATH)" $(PYTHON) -m src.cli chat
mcp:
	$(PYTHON) -m src.cli mcp
gpu:
	$(PYTHON) scripts/check_gpu.py
