#!/usr/bin/env bash
# Downloads the model weights at pinned revisions into ~/synergy_liars/models. Re-running resumes.
set -euo pipefail
cd ~/synergy_liars && source .venv/bin/activate
export HF_XET_HIGH_PERFORMANCE=1
get() { hf download "$1" --revision "$2" --local-dir "models/$3"; }
get Qwen/Qwen3-8B-AWQ            4da05a8edb55c6046cce958586c33b61da07bb79 qwen3-8b-awq   # 6.1 GB
get cyankiwi/Qwen3.5-4B-AWQ-4bit d3e78b50c727b1f34802f440ca84b25781625352 qwen35-4b-awq  # 4.1 GB
du -sh models/*
