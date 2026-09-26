#!/usr/bin/env python3
"""Run after pip install -e .; no torch dependency required for driver inspection."""

import json

from src.training.hardware import hardware

if __name__ == "__main__":
    print(json.dumps(hardware(), indent=2))
