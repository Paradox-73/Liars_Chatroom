#!/usr/bin/env bash
# Checks that vLLM runs on this card and driver with a small model of the same architecture as Qwen3.5/3.8.
set -euo pipefail
cd ~/synergy_liars && source .venv/bin/activate
export HF_XET_HIGH_PERFORMANCE=1
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/qwen35-0.8b
nvidia-smi --query-gpu=memory.free --format=csv
python scripts/server/smoke_test.py
