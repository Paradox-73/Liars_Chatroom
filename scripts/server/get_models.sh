#!/usr/bin/env bash
# Downloads the model weights at pinned revisions into ~/synergy_liars/models. Re-running resumes.
set -euo pipefail
cd ~/synergy_liars && source .venv/bin/activate
export HF_XET_HIGH_PERFORMANCE=1
get() { hf download "$1" --revision "$2" --local-dir "models/$3"; }
get Qwen/Qwen3-8B-AWQ            4da05a8edb55c6046cce958586c33b61da07bb79 qwen3-8b-awq   # 6.1 GB
get cyankiwi/Qwen3.5-9B-AWQ-4bit e4d2d39cede8c5d1716d824cd843e2889c648083 qwen35-9b-awq  # 9.1 GB
get cyankiwi/Qwen3.5-4B-AWQ-4bit d3e78b50c727b1f34802f440ca84b25781625352 qwen35-4b-awq  # 4.1 GB
get cyankiwi/Qwen3.8-27B-AWQ-INT4 e3a700906e6a73ba4290b1873dc2ba8198527ab5 qwen38-27b-awq # 21.0 GB, needs 30 GB free to load
du -sh models/*
